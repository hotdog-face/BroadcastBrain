"""The orchestrator: one command, one night, a finished library.

``bb run`` is this class. In order:

1. **Recover.** Reclaim leases from a previous run that was interrupted, and
   re-enqueue anything that was left mid-pipeline. This happens *first*, before
   any new work is discovered, so a resumed run finishes what it started.
2. **Discover.** Scan ``Incoming/`` for new material.
3. **Work.** Run the worker pool until the queue drains.
4. **Export.** Regenerate the Fieldstation export.
5. **Report.** Write a summary of what happened while you were asleep.

Then either exit, or -- with ``--watch`` -- sit on ``Incoming/`` and repeat
whenever something new appears.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import Settings
from ..db.base import Database, get_database
from ..db.queue import JobQueue
from ..db.repo import ClipRepository, EventRepository, RunRepository, SourceRepository
from ..db.schema import initialize
from ..library.layout import ensure_library_tree
from ..logging_setup import configure_logging
from ..types import ClipStatus, JobKind, Stage
from .discovery import Discovery, DiscoveryResult
from .reports import RunReport, write_report
from .worker import WorkerPool, run_until_drained

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """What one invocation accomplished."""

    run_id: Optional[int] = None
    discovery: Optional[DiscoveryResult] = None
    recovered_jobs: int = 0
    requeued_items: int = 0
    worker_stats: dict[str, Any] = field(default_factory=dict)
    clips_created: int = 0
    export_summary: Optional[dict[str, Any]] = None
    report_path: Optional[str] = None
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "discovery": self.discovery.as_dict() if self.discovery else None,
            "recovered_jobs": self.recovered_jobs,
            "requeued_items": self.requeued_items,
            "workers": self.worker_stats,
            "clips_created": self.clips_created,
            "export": self.export_summary,
            "report": self.report_path,
            "duration_s": round(self.duration_s, 1),
        }


class Pipeline:
    """The whole system, wired together."""

    def __init__(self, settings: Settings, *, db: Optional[Database] = None) -> None:
        self.settings = settings
        settings.paths.ensure()
        ensure_library_tree(settings.paths.require("library"))
        self.db = db or get_database(settings)
        self.queue = JobQueue(self.db)
        self.discovery = Discovery(settings, self.db, self.queue)
        self._pool: Optional[WorkerPool] = None

    # -- setup --------------------------------------------------------------

    def initialize(self) -> int:
        """Create or upgrade the database. Idempotent."""
        version = initialize(self.db)
        log.debug("database ready at schema version %s", version)
        return version

    # -- the main entry point ----------------------------------------------

    def run(
        self,
        *,
        scan: bool = True,
        export: bool = True,
        report: bool = True,
        on_tick: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> RunSummary:
        """Process everything and return. This is ``bb run``."""
        started = time.monotonic()
        self.initialize()
        summary = RunSummary()

        with self.db.write_session() as session:
            record = RunRepository.start(
                session,
                mode="run",
                host=socket.gethostname(),
                snapshot=self.settings.describe(),
            )
            summary.run_id = record.id

        clips_before = self._clip_count()

        summary.recovered_jobs = self.recover()
        summary.requeued_items = self.resume_unfinished()

        if scan:
            summary.discovery = self.discovery.scan()

        summary.worker_stats = run_until_drained(
            self.settings, self.db, self.queue, on_tick=on_tick
        )
        summary.clips_created = max(0, self._clip_count() - clips_before)

        if export and self.settings.export.enabled:
            summary.export_summary = self.export()

        summary.duration_s = time.monotonic() - started

        if report:
            run_report = self.build_report(summary)
            path = write_report(self.settings, run_report)
            summary.report_path = str(path)

        with self.db.write_session() as session:
            RunRepository.finish(
                session,
                summary.run_id or 0,
                sources_seen=summary.discovery.scanned if summary.discovery else 0,
                sources_completed=self._completed_source_count(),
                clips_created=summary.clips_created,
                jobs_completed=_as_int(summary.worker_stats.get("completed")),
                jobs_failed=_as_int(summary.worker_stats.get("dead")),
                report_path=summary.report_path,
            )

        log.info(
            "run finished in %.1f min: %d clip(s) created, %d job(s) completed",
            summary.duration_s / 60.0,
            summary.clips_created,
            _as_int(summary.worker_stats.get("completed")),
        )
        return summary

    # -- recovery -----------------------------------------------------------

    def recover(self) -> int:
        """Reclaim jobs whose worker died. Always safe, always first."""
        reclaimed = self.queue.reclaim_expired()
        if reclaimed:
            with self.db.write_session() as session:
                EventRepository.log(
                    session,
                    "recovered",
                    f"{reclaimed} job(s) reclaimed from an interrupted run",
                    level="warning",
                )
        return reclaimed

    def resume_unfinished(self) -> int:
        """Re-enqueue sources and clips that never reached the end.

        This is the counterpart to lease reclamation: leases cover work that
        was *running*, this covers work that was never queued because the
        process died between finishing one stage and enqueuing the next.
        """
        queued = 0
        with self.db.session() as session:
            sources = SourceRepository.unfinished(session)
            clips = ClipRepository.incomplete(session)
            source_jobs = [(s.id, Stage(s.stage)) for s in sources]
            clip_jobs = [(c.id, Stage(c.stage), ClipStatus(c.status)) for c in clips]

        for source_id, stage in source_jobs:
            source_kind = (
                JobKind.INGEST_SOURCE
                if stage is Stage.DISCOVERED
                else JobKind.SEGMENT_SOURCE
            )
            if self.queue.enqueue(source_kind, {"source_id": source_id}) is not None:
                queued += 1

        for clip_id, stage, status in clip_jobs:
            if status is ClipStatus.FAILED:
                continue
            clip_kind = _next_job_for_clip(stage)
            if clip_kind is None:
                continue
            if self.queue.enqueue(clip_kind, {"clip_id": clip_id}) is not None:
                queued += 1

        if queued:
            log.info("resumed %d unfinished item(s) from a previous run", queued)
        return queued

    # -- individual operations ---------------------------------------------

    def scan_incoming(self) -> DiscoveryResult:
        return self.discovery.scan()

    def export(self) -> dict[str, Any]:
        """Regenerate the Fieldstation export synchronously."""
        from ..export.fieldstation import FieldstationExporter

        exporter = FieldstationExporter(self.settings, self.db)
        return exporter.export().as_dict()

    def enqueue_export(self) -> None:
        self.queue.enqueue(JobKind.EXPORT_LIBRARY, {"scope": "library"}, requeue_if_done=True)

    def reanalyze(
        self,
        *,
        clip_ids: Optional[list[str]] = None,
        analyzers: Optional[list[str]] = None,
        only_missing: bool = False,
    ) -> int:
        """Queue clips for re-analysis.

        With ``only_missing`` this queues exactly the clips that lack a current
        result for the named analyzers -- the workflow for "I installed Whisper
        after last night's run".
        """
        from ..analyze.registry import all_analyzers, get
        from ..db.repo import AnalysisRepository

        targets: set[str] = set(clip_ids or ())

        if only_missing:
            names = analyzers or [a.name for a in all_analyzers()]
            with self.db.session() as session:
                for name in names:
                    analyzer = get(name)
                    version = analyzer.version if analyzer else None
                    targets.update(
                        AnalysisRepository.clips_missing_analyzer(
                            session, name, version=version, limit=100_000
                        )
                    )
        elif not targets:
            with self.db.session() as session:
                targets.update(str(clip.id) for clip in ClipRepository.iter_all(session))

        # `--missing` must *rely* on the cache, not override it. Forcing "*"
        # here re-ran every analyzer on every selected clip -- including
        # Whisper on clips that already had a perfectly good transcript --
        # which is the exact opposite of what the flag promises. The clips were
        # selected because something is missing; the runner's version check
        # then runs precisely that and skips the rest.
        payload_force: Optional[list[str]] = analyzers
        if payload_force is None and not only_missing:
            payload_force = ["*"]

        queued = 0
        for clip_id in sorted(targets):
            if (
                self.queue.enqueue(
                    JobKind.ANALYZE_CLIP,
                    {"clip_id": clip_id, "force": payload_force} if payload_force
                    else {"clip_id": clip_id},
                    requeue_if_done=True,
                )
                is not None
            ):
                queued += 1
        log.info("queued %d clip(s) for re-analysis", queued)
        return queued

    def recategorize(self) -> int:
        """Re-run classification over the whole library, from stored votes.

        Costs no model time at all: the votes were persisted when the analyzers
        ran. A change to the classification rules can be applied to a hundred
        thousand clips in seconds.
        """
        from ..categorize.classifier import Categorizer

        categorizer = Categorizer(self.settings, self.db)
        count = 0
        with self.db.session() as session:
            clip_ids = [str(clip.id) for clip in ClipRepository.iter_all(session)]
        for clip_id in clip_ids:
            if categorizer.categorize(clip_id) is not None:
                count += 1
        log.info("recategorized %d clip(s)", count)
        return count

    # -- reporting ----------------------------------------------------------

    def build_report(self, summary: RunSummary) -> RunReport:
        from .reports import collect_report

        return collect_report(self.settings, self.db, self.queue, summary)

    def status(self) -> dict[str, Any]:
        """A snapshot for ``bb status`` and the dashboard."""
        from ..dedupe.resolver import duplicate_statistics

        with self.db.session() as session:
            clip_stats = ClipRepository.statistics(session)
            source_counts = SourceRepository.counts_by_status(session)
        return {
            "queue": self.queue.counts_by_state(),
            "queue_by_kind": self.queue.counts_by_kind(),
            "sources": source_counts,
            "clips": clip_stats,
            "duplicates": duplicate_statistics(self.db),
            "settings": self.settings.describe(),
        }

    # -- helpers ------------------------------------------------------------

    def _clip_count(self) -> int:
        with self.db.session() as session:
            return int(ClipRepository.statistics(session)["total_clips"])

    def _completed_source_count(self) -> int:
        with self.db.session() as session:
            counts = SourceRepository.counts_by_status(session)
        return int(counts.get("complete", 0))


def _as_int(value: Any) -> int:
    """Worker statistics are a loosely typed dict; coerce at the boundary."""
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def _next_job_for_clip(stage: Stage) -> Optional[JobKind]:
    """Which job should run next for a clip sitting at ``stage``."""
    mapping = {
        Stage.SEGMENTED: JobKind.ANALYZE_CLIP,
        Stage.EXTRACTED: JobKind.ANALYZE_CLIP,
        Stage.ANALYZED: JobKind.FINGERPRINT_CLIP,
        Stage.FINGERPRINTED: JobKind.DEDUPE_CLIP,
        Stage.DEDUPED: JobKind.CATEGORIZE_CLIP,
        Stage.CATEGORIZED: JobKind.MATERIALIZE_CLIP,
    }
    return mapping.get(stage)


def build_pipeline(settings: Settings, *, configure_logs: bool = True) -> Pipeline:
    """Construct a ready-to-use pipeline. Used by the CLI and the dashboard."""
    if configure_logs:
        configure_logging(settings)
    pipeline = Pipeline(settings)
    pipeline.initialize()
    return pipeline
