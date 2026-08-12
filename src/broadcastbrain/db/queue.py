"""Durable, resumable work queue.

Every unit of work lives in the ``jobs`` table, so the queue survives process
death by construction. Three properties make the "run it overnight and walk
away" promise real:

1. **Idempotent enqueue.** Each job carries a ``dedupe_key`` derived from what
   it will do. Enqueueing the same work twice is a no-op, so a rescan of
   Incoming never duplicates effort.
2. **Leases, not locks.** A claimed job gets a lease with an expiry. A worker
   that is killed leaves an expired lease, and the next run reclaims it. There
   is no "stuck running forever" state that needs manual cleanup.
3. **Typed failure.** Permanent failures go straight to ``dead``; transient
   ones get exponential backoff via ``available_at``. Attempts are never burned
   on a file that can never decode.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional, Sequence

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..constants import DEFAULT_MAX_ATTEMPTS, JOB_LEASE_SECONDS, JOB_PRIORITY
from ..types import JobKind, JobState
from ..util.backoff import backoff_delay
from ..util.hashing import stable_id
from .base import Database
from .models import Job, utcnow

log = logging.getLogger(__name__)


def make_worker_id(suffix: Optional[str] = None) -> str:
    """Stable-ish identifier so an operator can tell which box owns a lease."""
    host = socket.gethostname().split(".")[0]
    tail = suffix or threading.current_thread().name
    return f"{host}:{os.getpid()}:{tail}"[:64]


def default_dedupe_key(kind: JobKind, payload: dict[str, Any]) -> str:
    """Derive the identity of a job from its kind and its salient payload keys.

    Only identity-bearing keys participate, so re-enqueueing the same clip with
    a different ``reason`` still collapses to one job.
    """
    identity_keys = ("source_id", "clip_id", "path", "identifier", "scope", "analyzer")
    parts = [kind.value] + [f"{k}={payload[k]}" for k in identity_keys if k in payload]
    return stable_id(*parts)


class JobQueue:
    """Database-backed queue. All methods are safe to call from many threads."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # -- producing ----------------------------------------------------------

    def enqueue(
        self,
        kind: JobKind,
        payload: Optional[dict[str, Any]] = None,
        *,
        priority: Optional[int] = None,
        needs_gpu: bool = False,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        dedupe_key: Optional[str] = None,
        available_at: Optional[datetime] = None,
        requeue_if_done: bool = False,
        session: Optional[Session] = None,
    ) -> Optional[int]:
        """Add a job unless an equivalent one already exists.

        Args:
            requeue_if_done: Reset a previously completed job back to pending.
                This is what ``bb reanalyze`` uses to force re-work.

        Returns:
            The job id, or ``None`` when the job already existed and was left
            alone.
        """
        payload = dict(payload or {})
        key = dedupe_key or default_dedupe_key(kind, payload)
        resolved_priority = priority if priority is not None else JOB_PRIORITY.get(kind.value, 0)

        def _do(sess: Session) -> Optional[int]:
            existing = sess.execute(select(Job).where(Job.dedupe_key == key)).scalar_one_or_none()
            if existing is not None:
                if existing.state in (JobState.PENDING.value, JobState.RUNNING.value):
                    return None
                if not requeue_if_done and existing.state == JobState.DONE.value:
                    return None
                existing.state = JobState.PENDING.value
                existing.attempts = 0
                existing.payload = payload
                existing.priority = resolved_priority
                existing.needs_gpu = needs_gpu
                existing.max_attempts = max_attempts
                existing.available_at = available_at or utcnow()
                existing.lease_expires_at = None
                existing.worker_id = None
                existing.last_error = None
                existing.finished_at = None
                existing.progress = 0.0
                sess.flush()
                return int(existing.id)

            job = Job(
                kind=kind.value,
                dedupe_key=key,
                payload=payload,
                state=JobState.PENDING.value,
                priority=resolved_priority,
                needs_gpu=needs_gpu,
                max_attempts=max_attempts,
                available_at=available_at or utcnow(),
            )
            sess.add(job)
            try:
                sess.flush()
            except IntegrityError:
                # Lost a race with another worker enqueueing the same key.
                sess.rollback()
                return None
            return int(job.id)

        if session is not None:
            return _do(session)
        with self.db.write_session() as sess:
            return _do(sess)

    def enqueue_many(
        self, items: Iterable[tuple[JobKind, dict[str, Any]]], *, needs_gpu: bool = False
    ) -> int:
        """Bulk enqueue in one transaction. Returns how many were new."""
        created = 0
        with self.db.write_session() as session:
            for kind, payload in items:
                if self.enqueue(kind, payload, needs_gpu=needs_gpu, session=session) is not None:
                    created += 1
        return created

    # -- consuming ----------------------------------------------------------

    def claim(
        self,
        worker_id: str,
        *,
        kinds: Optional[Sequence[JobKind]] = None,
        allow_gpu: bool = True,
        gpu_only: bool = False,
        lease_seconds: int = JOB_LEASE_SECONDS,
    ) -> Optional[Job]:
        """Atomically take the highest-priority runnable job.

        Args:
            allow_gpu: When False, GPU jobs are left for GPU workers.
            gpu_only: When True, only GPU jobs are considered.
        """
        now = utcnow()
        with self.db.write_session() as session:
            stmt = select(Job).where(
                Job.state == JobState.PENDING.value,
                Job.available_at <= now,
            )
            if kinds:
                stmt = stmt.where(Job.kind.in_([k.value for k in kinds]))
            if gpu_only:
                stmt = stmt.where(Job.needs_gpu.is_(True))
            elif not allow_gpu:
                stmt = stmt.where(Job.needs_gpu.is_(False))
            stmt = stmt.order_by(Job.priority.desc(), Job.id.asc()).limit(1)

            if not self.db.is_sqlite:
                # PostgreSQL: let concurrent workers skip each other's rows.
                stmt = stmt.with_for_update(skip_locked=True)

            job = session.execute(stmt).scalar_one_or_none()
            if job is None:
                return None

            # Guarded update: on SQLite the surrounding BEGIN IMMEDIATE already
            # serializes us, but the state check keeps this correct anywhere.
            result = session.execute(
                update(Job)
                .where(Job.id == job.id, Job.state == JobState.PENDING.value)
                .values(
                    state=JobState.RUNNING.value,
                    worker_id=worker_id,
                    attempts=Job.attempts + 1,
                    started_at=now,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    progress=0.0,
                    progress_note=None,
                )
            )
            if _rowcount(result) != 1:  # pragma: no cover - lost race
                return None
            session.flush()
            session.refresh(job)
            session.expunge(job)
            return job

    def heartbeat(
        self,
        job_id: int,
        *,
        lease_seconds: int = JOB_LEASE_SECONDS,
        progress: Optional[float] = None,
        note: Optional[str] = None,
    ) -> None:
        """Extend the lease of a long-running job and publish progress.

        A 6-hour movie transcription must not have its lease expire underneath
        it while it is making perfectly good progress.
        """
        values: dict[str, Any] = {
            "lease_expires_at": utcnow() + timedelta(seconds=lease_seconds)
        }
        if progress is not None:
            values["progress"] = max(0.0, min(1.0, progress))
        if note is not None:
            values["progress_note"] = note[:256]
        with self.db.write_session() as session:
            session.execute(update(Job).where(Job.id == job_id).values(**values))

    def complete(self, job_id: int, *, note: Optional[str] = None) -> None:
        with self.db.write_session() as session:
            session.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(
                    state=JobState.DONE.value,
                    finished_at=utcnow(),
                    lease_expires_at=None,
                    progress=1.0,
                    progress_note=note,
                    last_error=None,
                )
            )

    def fail(self, job_id: int, error: str, *, permanent: bool = False) -> JobState:
        """Record a failure and decide whether it is retried.

        Returns:
            The resulting state, so the caller can log "will retry" vs "dead".
        """
        with self.db.write_session() as session:
            job = session.get(Job, job_id)
            if job is None:
                return JobState.DEAD
            exhausted = job.attempts >= job.max_attempts
            if permanent or exhausted:
                job.state = JobState.DEAD.value
                job.finished_at = utcnow()
                job.lease_expires_at = None
            else:
                job.state = JobState.PENDING.value
                job.available_at = utcnow() + timedelta(seconds=backoff_delay(job.attempts))
                job.lease_expires_at = None
                job.worker_id = None
            job.last_error = error[:8000]
            session.flush()
            return JobState(job.state)

    def cancel(self, job_id: int) -> None:
        with self.db.write_session() as session:
            session.execute(
                update(Job)
                .where(Job.id == job_id, Job.state.in_([JobState.PENDING.value, JobState.RUNNING.value]))
                .values(state=JobState.CANCELLED.value, finished_at=utcnow(), lease_expires_at=None)
            )

    # -- maintenance --------------------------------------------------------

    def reclaim_expired(self) -> int:
        """Return jobs whose worker vanished to the pending pool.

        Called on every start. This is what makes ``Ctrl-C`` at 3 a.m. cost
        nothing but the work that was literally in flight.
        """
        now = utcnow()
        with self.db.write_session() as session:
            result = session.execute(
                update(Job)
                .where(
                    Job.state == JobState.RUNNING.value,
                    Job.lease_expires_at.is_not(None),
                    Job.lease_expires_at < now,
                )
                .values(
                    state=JobState.PENDING.value,
                    worker_id=None,
                    lease_expires_at=None,
                    available_at=now,
                    last_error="lease expired; worker did not finish",
                )
            )
            count = _rowcount(result)
        if count:
            log.warning("reclaimed %d job(s) with expired leases", count)
        return count

    def requeue_dead(self, *, kinds: Optional[Sequence[JobKind]] = None) -> int:
        """Give up-for-dead jobs another chance (``bb retry``)."""
        with self.db.write_session() as session:
            stmt = update(Job).where(Job.state == JobState.DEAD.value)
            if kinds:
                stmt = stmt.where(Job.kind.in_([k.value for k in kinds]))
            result = session.execute(
                stmt.values(
                    state=JobState.PENDING.value,
                    attempts=0,
                    available_at=utcnow(),
                    last_error=None,
                    lease_expires_at=None,
                    worker_id=None,
                )
            )
            return _rowcount(result)

    def purge_completed(self, *, older_than_days: int = 30) -> int:
        """Trim finished jobs so the queue table stays small at 100 TB scale."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        with self.db.write_session() as session:
            result = session.execute(
                delete(Job).where(Job.state == JobState.DONE.value, Job.finished_at < cutoff)
            )
            return _rowcount(result)

    # -- introspection ------------------------------------------------------

    def counts_by_state(self) -> dict[str, int]:
        with self.db.session() as session:
            rows = session.execute(
                select(Job.state, func.count()).group_by(Job.state)
            ).all()
        counts = {state.value: 0 for state in JobState}
        counts.update({str(state): int(count) for state, count in rows})
        return counts

    def counts_by_kind(self) -> dict[str, dict[str, int]]:
        with self.db.session() as session:
            rows = session.execute(
                select(Job.kind, Job.state, func.count()).group_by(Job.kind, Job.state)
            ).all()
        out: dict[str, dict[str, int]] = {}
        for kind, state, count in rows:
            out.setdefault(str(kind), {})[str(state)] = int(count)
        return out

    def pending_count(self, *, include_running: bool = True) -> int:
        states = [JobState.PENDING.value]
        if include_running:
            states.append(JobState.RUNNING.value)
        with self.db.session() as session:
            return int(
                session.execute(
                    select(func.count()).select_from(Job).where(Job.state.in_(states))
                ).scalar_one()
            )

    def has_runnable_work(self, *, horizon_s: float = 60.0) -> bool:
        """True when a worker could pick something up now or very soon.

        The horizon matters: a job that failed transiently is pending with an
        ``available_at`` up to fifteen minutes in the future. Counting that as
        "runnable" would keep a draining run alive, doing nothing, until the
        backoff elapsed. Anything beyond the horizon is left for the next run,
        which re-enqueues it during recovery.
        """
        cutoff = utcnow() + timedelta(seconds=horizon_s)
        with self.db.session() as session:
            return (
                session.execute(
                    select(Job.id)
                    .where(
                        or_(
                            Job.state == JobState.RUNNING.value,
                            and_(
                                Job.state == JobState.PENDING.value,
                                Job.available_at <= cutoff,
                            ),
                        )
                    )
                    .limit(1)
                ).first()
                is not None
            )

    def running_jobs(self, limit: int = 50) -> list[Job]:
        with self.db.session() as session:
            jobs = list(
                session.execute(
                    select(Job)
                    .where(Job.state == JobState.RUNNING.value)
                    .order_by(Job.started_at.asc())
                    .limit(limit)
                ).scalars()
            )
            for job in jobs:
                session.expunge(job)
            return jobs

    def recent_failures(self, limit: int = 25) -> list[Job]:
        with self.db.session() as session:
            jobs = list(
                session.execute(
                    select(Job)
                    .where(Job.state.in_([JobState.DEAD.value, JobState.FAILED.value]))
                    .order_by(Job.updated_at.desc())
                    .limit(limit)
                ).scalars()
            )
            for job in jobs:
                session.expunge(job)
            return jobs


def _rowcount(result: Any) -> int:
    """Rows affected by a DML statement.

    ``rowcount`` is declared on ``CursorResult``, not on the generic ``Result``
    that ``Session.execute`` is typed as returning, so it is read defensively
    here rather than cast at a dozen call sites.
    """
    return int(getattr(result, "rowcount", 0) or 0)
