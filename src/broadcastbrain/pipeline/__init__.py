"""The pipeline: discovery, stages, workers, orchestration and watching."""

from __future__ import annotations

from .discovery import Discovery, DiscoveryResult, mark_source_complete
from .orchestrator import Pipeline, RunSummary, build_pipeline
from .reports import RunReport, collect_report, latest_report, render_markdown, write_report
from .stages import HANDLERS, StageContext, get_handler
from .watcher import Watcher, WatchState
from .worker import Worker, WorkerPool, WorkerStats, run_until_drained

__all__ = [
    "HANDLERS",
    "Discovery",
    "DiscoveryResult",
    "Pipeline",
    "RunReport",
    "RunSummary",
    "StageContext",
    "WatchState",
    "Watcher",
    "Worker",
    "WorkerPool",
    "WorkerStats",
    "build_pipeline",
    "collect_report",
    "get_handler",
    "latest_report",
    "mark_source_complete",
    "render_markdown",
    "run_until_drained",
    "write_report",
]
