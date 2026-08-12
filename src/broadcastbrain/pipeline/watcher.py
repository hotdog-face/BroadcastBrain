"""Watching ``Incoming/`` for new material.

Polling, not inotify/FSEvents. Three reasons, in order of importance: media is
routinely dropped onto network shares where OS-level watches do not fire at
all; a poll loop has no platform-specific code to go wrong at 4 a.m.; and the
latency that matters here is minutes, not milliseconds.

The watcher owns a long-lived worker pool rather than starting one per batch,
so a file dropped while another is processing is picked up immediately instead
of waiting for the current batch to drain.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import Settings
from ..constants import WATCH_POLL_SECONDS
from ..db.repo import EventRepository
from ..types import JobKind
from .orchestrator import Pipeline
from .worker import WorkerPool

log = logging.getLogger(__name__)


@dataclass
class WatchState:
    """Live state of a watch session, surfaced by the dashboard."""

    started_at: float = field(default_factory=time.time)
    scans: int = 0
    sources_found: int = 0
    last_scan_at: Optional[float] = None
    last_export_at: Optional[float] = None
    stopping: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "scans": self.scans,
            "sources_found": self.sources_found,
            "last_scan_at": self.last_scan_at,
            "last_export_at": self.last_export_at,
            "stopping": self.stopping,
        }


class Watcher:
    """Runs the pipeline continuously, reacting to new files."""

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        poll_seconds: float = WATCH_POLL_SECONDS,
        export_after_idle_s: float = 300.0,
    ) -> None:
        self.pipeline = pipeline
        self.settings: Settings = pipeline.settings
        self.poll_seconds = poll_seconds
        #: Re-export once the queue has been quiet this long. Exporting after
        #: every single clip would rewrite the whole export constantly.
        self.export_after_idle_s = export_after_idle_s
        self.state = WatchState()
        self._stop = threading.Event()
        self._pool: Optional[WorkerPool] = None

    # -- lifecycle ----------------------------------------------------------

    def install_signal_handlers(self) -> None:
        """Stop cleanly on Ctrl-C / SIGTERM, letting in-flight jobs finish."""

        def _handler(signum: int, _frame: Any) -> None:  # pragma: no cover - signal path
            log.warning("signal %s received; finishing in-flight work", signum)
            self.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                log.debug("could not install handler for signal %s", sig)

    def stop(self) -> None:
        self.state.stopping = True
        self._stop.set()

    # -- main loop ----------------------------------------------------------

    def run(self, *, on_tick: Optional[Callable[[dict[str, Any]], None]] = None) -> WatchState:
        """Watch until stopped. This is ``bb watch``."""
        self.pipeline.initialize()
        self.pipeline.recover()
        self.pipeline.resume_unfinished()

        self._pool = WorkerPool(self.settings, self.pipeline.db, self.pipeline.queue)
        # drain=False: workers wait for work instead of exiting when idle.
        self._pool.start(drain=False)

        log.info(
            "watching %s every %.0fs -- drop files in and they will be processed",
            self.settings.paths.require("incoming"), self.poll_seconds,
        )

        idle_since: Optional[float] = None
        try:
            while not self._stop.is_set():
                result = self.pipeline.scan_incoming()
                self.state.scans += 1
                self.state.sources_found += result.new_sources
                self.state.last_scan_at = time.time()

                if result.new_sources:
                    log.info("watcher: %s", result.summary())
                    idle_since = None

                pending = self.pipeline.queue.pending_count()
                if pending == 0:
                    idle_since = idle_since or time.time()
                    if (
                        self.settings.export.enabled
                        and time.time() - idle_since >= self.export_after_idle_s
                        and self.state.last_export_at != idle_since
                    ):
                        self._export(idle_since)
                else:
                    idle_since = None

                if on_tick is not None:
                    snapshot = self.state.as_dict()
                    snapshot["queue_pending"] = pending
                    if self._pool is not None:
                        snapshot["workers"] = self._pool.snapshot()
                    on_tick(snapshot)

                self._stop.wait(self.poll_seconds)
        except KeyboardInterrupt:  # pragma: no cover - operator interrupt
            log.warning("interrupted; shutting down cleanly")
        finally:
            self._shutdown()
        return self.state

    def _export(self, idle_marker: float) -> None:
        """Refresh the Fieldstation export after a quiet spell."""
        log.info("queue idle; refreshing the Fieldstation export")
        self.pipeline.queue.enqueue(
            JobKind.EXPORT_LIBRARY, {"scope": "library"}, requeue_if_done=True
        )
        self.state.last_export_at = idle_marker

    def _shutdown(self) -> None:
        if self._pool is not None:
            log.info("stopping workers (in-flight jobs will finish)")
            self._pool.stop(timeout=120.0)
            self._pool = None
        with self.pipeline.db.write_session() as session:
            EventRepository.log(
                session,
                "watch_stopped",
                f"watched for {self.state.as_dict()['uptime_s']}s "
                f"across {self.state.scans} scan(s)",
            )
        log.info("watcher stopped after %d scan(s)", self.state.scans)
