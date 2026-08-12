"""Worker threads.

Threads rather than processes, because essentially all of the wall-clock time
in this pipeline is spent inside ffmpeg subprocesses or model libraries that
release the GIL. Threads also share one loaded Whisper model instead of one per
process, which on a 16-core box is the difference between 500 MB and 8 GB of
resident memory.

Two pools run concurrently:

* **CPU workers** take anything that does not need the GPU.
* **GPU workers** take GPU jobs, bounded by ``workers.gpu_workers`` so a single
  card is never oversubscribed while twelve CPU workers keep running.

A heartbeat thread renews leases for jobs that are still making progress, so a
six-hour transcription is never reclaimed out from under itself.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..config import Settings
from ..constants import QUEUE_IDLE_POLL_S
from ..db.base import Database
from ..db.models import Job
from ..db.queue import JobQueue, make_worker_id
from ..db.repo import EventRepository
from ..errors import PermanentError, RetryableError
from ..logging_setup import log_context
from ..types import JobState
from .stages import StageContext, get_handler

log = logging.getLogger(__name__)

#: How often a running job renews its lease.
HEARTBEAT_INTERVAL_S = 60.0


@dataclass
class WorkerStats:
    completed: int = 0
    failed: int = 0
    dead: int = 0
    busy_seconds: float = 0.0
    current: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "completed": self.completed,
            "failed": self.failed,
            "dead": self.dead,
            "busy_seconds": round(self.busy_seconds, 1),
            "current": dict(self.current),
        }


class Worker(threading.Thread):
    """One worker thread: claim, run, record, repeat."""

    def __init__(
        self,
        name: str,
        settings: Settings,
        db: Database,
        queue: JobQueue,
        stop_event: threading.Event,
        stats: WorkerStats,
        stats_lock: threading.Lock,
        *,
        gpu_only: bool = False,
        allow_gpu: bool = True,
        drain: bool = True,
        idle_poll_s: float = QUEUE_IDLE_POLL_S,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.settings = settings
        self.db = db
        self.queue = queue
        self.stop_event = stop_event
        self.stats = stats
        self.stats_lock = stats_lock
        self.gpu_only = gpu_only
        self.allow_gpu = allow_gpu
        #: When True the worker exits once the queue is empty. When False it
        #: waits for new work -- which is what ``bb watch`` needs.
        self.drain = drain
        self.idle_poll_s = idle_poll_s
        self.worker_id = make_worker_id(name)

    def run(self) -> None:
        log.debug("worker %s started", self.name)
        idle_rounds = 0

        while not self.stop_event.is_set():
            job = self.queue.claim(
                self.worker_id,
                allow_gpu=self.allow_gpu,
                gpu_only=self.gpu_only,
                lease_seconds=self.settings.workers.lease_seconds,
            )
            if job is None:
                idle_rounds += 1
                if self.drain and idle_rounds >= 2 and not self.queue.has_runnable_work():
                    break
                self.stop_event.wait(self.idle_poll_s)
                continue

            idle_rounds = 0
            self._execute(job)

        log.debug("worker %s finished", self.name)

    # -- execution ----------------------------------------------------------

    def _execute(self, job: Job) -> None:
        handler = get_handler(job.kind)
        if handler is None:
            self.queue.fail(job.id, f"no handler for job kind {job.kind}", permanent=True)
            return

        started = time.monotonic()
        heartbeat = _Heartbeat(self.queue, job.id, self.settings.workers.lease_seconds)
        ctx = StageContext(
            settings=self.settings,
            db=self.db,
            queue=self.queue,
            progress=heartbeat.progress,
        )

        with log_context(job=job.id, kind=job.kind, worker=self.name):
            self._set_current(f"{job.kind} #{job.id}")
            heartbeat.start()
            try:
                handler(ctx, job)
            except PermanentError as exc:
                self._record_failure(job, str(exc), permanent=True)
            except RetryableError as exc:
                self._record_failure(job, str(exc), permanent=False)
            except KeyboardInterrupt:  # pragma: no cover - operator interrupt
                self.queue.fail(job.id, "interrupted", permanent=False)
                self.stop_event.set()
                raise
            except Exception as exc:
                log.exception("job %s (%s) raised", job.id, job.kind)
                self._record_failure(job, f"{type(exc).__name__}: {exc}", permanent=False)
            else:
                self.queue.complete(job.id)
                with self.stats_lock:
                    self.stats.completed += 1
            finally:
                heartbeat.stop()
                elapsed = time.monotonic() - started
                with self.stats_lock:
                    self.stats.busy_seconds += elapsed
                self._set_current(None)

    def _record_failure(self, job: Job, message: str, *, permanent: bool) -> None:
        state = self.queue.fail(job.id, message, permanent=permanent)
        with self.stats_lock:
            if state is JobState.DEAD:
                self.stats.dead += 1
            else:
                self.stats.failed += 1

        level = "error" if state is JobState.DEAD else "warning"
        log.log(
            logging.ERROR if state is JobState.DEAD else logging.WARNING,
            "job %s (%s) %s: %s",
            job.id, job.kind, "died" if state is JobState.DEAD else "will retry", message,
        )
        with self.db.write_session() as session:
            EventRepository.log(
                session,
                "job_failed",
                f"{job.kind}: {message}",
                level=level,
                job_id=job.id,
                clip_id=job.payload.get("clip_id"),
                source_id=job.payload.get("source_id"),
            )

    def _set_current(self, description: Optional[str]) -> None:
        with self.stats_lock:
            if description is None:
                self.stats.current.pop(self.name, None)
            else:
                self.stats.current[self.name] = description


class _Heartbeat:
    """Keeps a running job's lease alive and publishes its progress."""

    def __init__(self, queue: JobQueue, job_id: int, lease_seconds: int) -> None:
        self._queue = queue
        self._job_id = job_id
        self._lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fraction = 0.0
        self._note = ""
        self._lock = threading.Lock()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name=f"heartbeat-{self._job_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def progress(self, fraction: float, note: str) -> None:
        """Called by stage handlers; cheap, in-memory, flushed by the loop."""
        with self._lock:
            self._fraction = fraction
            self._note = note

    def _loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_INTERVAL_S):
            with self._lock:
                fraction, note = self._fraction, self._note
            try:
                self._queue.heartbeat(
                    self._job_id,
                    lease_seconds=self._lease_seconds,
                    progress=fraction,
                    note=note,
                )
            except Exception as exc:
                log.debug("heartbeat failed for job %s: %s", self._job_id, exc)


class WorkerPool:
    """Owns the worker threads for one run."""

    def __init__(self, settings: Settings, db: Database, queue: JobQueue) -> None:
        self.settings = settings
        self.db = db
        self.queue = queue
        self.stop_event = threading.Event()
        self.stats = WorkerStats()
        self._stats_lock = threading.Lock()
        self._workers: list[Worker] = []

    def start(self, *, drain: bool = True) -> None:
        """Spin up CPU and GPU workers."""
        cpu_count = self.settings.workers.resolved_cpu_workers()
        gpu_count = max(0, self.settings.workers.gpu_workers)

        for index in range(cpu_count):
            self._workers.append(
                Worker(
                    f"cpu-{index + 1}",
                    self.settings, self.db, self.queue, self.stop_event,
                    self.stats, self._stats_lock,
                    allow_gpu=(gpu_count == 0),  # no GPU pool => CPU takes it all
                    drain=drain,
                )
            )
        for index in range(gpu_count):
            self._workers.append(
                Worker(
                    f"gpu-{index + 1}",
                    self.settings, self.db, self.queue, self.stop_event,
                    self.stats, self._stats_lock,
                    gpu_only=True,
                    drain=drain,
                )
            )

        for worker in self._workers:
            worker.start()
        log.info("started %d CPU worker(s) and %d GPU worker(s)", cpu_count, gpu_count)

    def wait(self, *, timeout: Optional[float] = None) -> bool:
        """Block until every worker exits. Returns True if all finished."""
        deadline = time.monotonic() + timeout if timeout else None
        for worker in self._workers:
            remaining = None if deadline is None else max(0.1, deadline - time.monotonic())
            worker.join(timeout=remaining)
        return not any(worker.is_alive() for worker in self._workers)

    def stop(self, *, timeout: float = 30.0) -> None:
        """Ask workers to finish their current job and exit.

        In-flight jobs are *not* killed. They finish, or their leases expire and
        the next run picks them up -- either way no work is lost or corrupted.
        """
        self.stop_event.set()
        self.wait(timeout=timeout)

    def snapshot(self) -> dict[str, object]:
        with self._stats_lock:
            stats = self.stats.as_dict()
        stats["workers"] = len(self._workers)
        stats["alive"] = sum(1 for w in self._workers if w.is_alive())
        return stats

    @property
    def running(self) -> bool:
        return any(worker.is_alive() for worker in self._workers)


def run_until_drained(
    settings: Settings,
    db: Database,
    queue: JobQueue,
    *,
    on_tick: Optional[Callable[[dict[str, object]], None]] = None,
    tick_interval_s: float = 5.0,
) -> dict[str, object]:
    """Process everything currently queued, then return.

    This is the body of ``bb run``: start the pool, report progress while it
    works, and hand back the statistics for the nightly report.
    """
    pool = WorkerPool(settings, db, queue)
    pool.start(drain=True)
    try:
        while pool.running:
            time.sleep(tick_interval_s)
            if on_tick is not None:
                on_tick(pool.snapshot())
    except KeyboardInterrupt:  # pragma: no cover - operator interrupt
        log.warning("interrupted; letting in-flight jobs finish (Ctrl-C again to force)")
        pool.stop(timeout=60.0)
        raise
    pool.wait()
    return pool.snapshot()
