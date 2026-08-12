"""The download manager: search, queue, fetch, hand off to the pipeline.

Downloads are tracked in the database exactly like pipeline work, for the same
reason: a 12 GB fetch that dies at 90% should resume, not restart, and an item
already downloaded should never be downloaded again -- even across runs, even
if the file has since been ingested and moved out of ``Incoming/``.

The ``(source_name, identifier, file_name)`` uniqueness constraint on the
downloads table is what enforces that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy import func, select

from ..config import Settings
from ..db.base import Database
from ..db.models import Download
from ..db.queue import JobQueue
from ..db.repo import EventRepository
from ..errors import DownloadError
from ..types import JobKind
from .base import ArchiveSource, RemoteItem, SearchSpec, all_sources, get_source

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class FetchPlan:
    """What a search decided to download, before anything is fetched."""

    queued: list[dict[str, Any]] = field(default_factory=list)
    skipped_existing: int = 0
    skipped_format: int = 0
    skipped_size: int = 0
    skipped_budget: int = 0
    total_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "queued": len(self.queued),
            "skipped_existing": self.skipped_existing,
            "skipped_format": self.skipped_format,
            "skipped_size": self.skipped_size,
            "skipped_budget": self.skipped_budget,
            "total_gb": round(self.total_bytes / (1024**3), 2),
            "items": self.queued[:50],
        }

    def summary(self) -> str:
        return (
            f"{len(self.queued)} item(s) queued ({self.total_bytes / (1024**3):.1f} GB); "
            f"skipped {self.skipped_existing} already known, "
            f"{self.skipped_format} unusable format, {self.skipped_size} too small, "
            f"{self.skipped_budget} over budget"
        )


class DownloadManager:
    """Coordinates archive sources with the download queue."""

    def __init__(self, settings: Settings, db: Database, queue: JobQueue) -> None:
        self.settings = settings
        self.db = db
        self.queue = queue
        self.destination = settings.paths.require("downloads")

    # -- planning -----------------------------------------------------------

    def plan(
        self,
        queries: Optional[list[str]] = None,
        *,
        source_name: str = "internet_archive",
        limit: Optional[int] = None,
    ) -> FetchPlan:
        """Search and queue everything worth downloading.

        Nothing is fetched here. The plan is queued as jobs, so downloads run
        through the same worker pool, retries and progress reporting as the
        rest of the pipeline.
        """
        source = get_source(source_name)
        if source is None:
            raise DownloadError(f"unknown archive source: {source_name}")

        config = self.settings.archive
        search_terms = queries or config.queries
        if not search_terms:
            raise DownloadError(
                "no search queries configured; set archive.queries or pass them on the command line"
            )

        plan = FetchPlan()
        budget_bytes = int(config.max_total_gb * (1024**3))
        spent = self._downloaded_bytes()

        for query in search_terms:
            spec = SearchSpec(
                query=query,
                limit=limit or config.max_items_per_query,
                collections=list(config.collections),
                date_range=config.date_range,
            )
            for item in source.search(spec):
                detailed = source.fetch_item(item.identifier) or item
                self._consider(source, detailed, query, plan, budget_bytes, spent)
                spent = self._downloaded_bytes() + plan.total_bytes

        log.info("fetch plan: %s", plan.summary())
        return plan

    def _consider(
        self,
        source: ArchiveSource,
        item: RemoteItem,
        query: str,
        plan: FetchPlan,
        budget_bytes: int,
        spent_bytes: int,
    ) -> None:
        """Decide whether one item earns a download slot."""
        config = self.settings.archive
        file = item.best_file(list(config.accept_formats))
        if file is None:
            plan.skipped_format += 1
            return
        if file.size_mb < config.min_item_mb:
            plan.skipped_size += 1
            return
        if spent_bytes + file.size_bytes > budget_bytes:
            plan.skipped_budget += 1
            return

        with self.db.session() as session:
            existing = session.execute(
                select(Download).where(
                    Download.source_name == source.name,
                    Download.identifier == item.identifier,
                    Download.file_name == file.name,
                )
            ).scalar_one_or_none()
        if existing is not None:
            plan.skipped_existing += 1
            return

        with self.db.write_session() as session:
            session.add(
                Download(
                    source_name=source.name,
                    identifier=item.identifier,
                    file_name=file.name,
                    url=file.url,
                    title=item.title,
                    query=query,
                    state="pending",
                    size_bytes=file.size_bytes,
                    item_metadata=item.as_dict(),
                )
            )
        self.queue.enqueue(
            JobKind.DOWNLOAD_ITEM,
            {
                "source_name": source.name,
                "identifier": item.identifier,
                "file_name": file.name,
            },
        )
        plan.queued.append(
            {
                "identifier": item.identifier,
                "title": item.title,
                "file": file.name,
                "size_mb": round(file.size_mb, 1),
                "year": item.year,
            }
        )
        plan.total_bytes += file.size_bytes

    # -- fetching -----------------------------------------------------------

    def download_one(
        self,
        *,
        source_name: str,
        identifier: str,
        file_name: Optional[str] = None,
        progress: Optional[ProgressFn] = None,
    ) -> Optional[Path]:
        """Fetch one queued item. Called by the ``download_item`` stage."""
        source = get_source(source_name)
        if source is None:
            raise DownloadError(f"unknown archive source: {source_name}")

        with self.db.write_session() as session:
            record = self._record(session, source_name, identifier, file_name)
            if record is None:
                raise DownloadError(f"no download record for {identifier}")
            if record.state == "complete" and record.local_path:
                existing = Path(record.local_path)
                if existing.exists():
                    log.debug("%s already downloaded", identifier)
                    return existing
            record.state = "downloading"
            record.attempts += 1

        item = source.fetch_item(identifier)
        if item is None:
            self._mark_failed(source_name, identifier, file_name, "item metadata unavailable")
            raise DownloadError(f"could not fetch metadata for {identifier}")

        file = next((f for f in item.files if f.name == file_name), None) or item.best_file(
            list(self.settings.archive.accept_formats)
        )
        if file is None:
            self._mark_failed(source_name, identifier, file_name, "no acceptable file in item")
            raise DownloadError(f"no usable file in {identifier}")

        target = self.destination / _safe_name(identifier, file.name)
        try:
            path = source.download(item, file, target, progress=progress)
        except DownloadError as exc:
            self._mark_failed(source_name, identifier, file.name, str(exc))
            raise

        size = path.stat().st_size if path.exists() else 0
        with self.db.write_session() as session:
            record = self._record(session, source_name, identifier, file.name)
            if record is not None:
                record.state = "complete"
                record.local_path = str(path)
                record.downloaded_bytes = size
                record.error = None
                from ..db.models import utcnow

                record.completed_at = utcnow()
            EventRepository.log(
                session,
                "downloaded",
                f"{item.title} ({size / (1024**2):.0f} MB)",
                detail={"identifier": identifier, "path": str(path)},
            )

        if self.settings.archive.auto_queue:
            self._hand_to_pipeline(path)
        return path

    def _hand_to_pipeline(self, path: Path) -> None:
        """Move a finished download into ``Incoming/`` so discovery picks it up.

        Going through Incoming rather than registering the source directly
        keeps one code path for ingestion -- the downloader gets the same
        stability checks, identity checks and rejection handling as a file
        dropped by hand.
        """
        from ..util.atomic import atomic_move

        incoming = self.settings.paths.require("incoming")
        try:
            moved = atomic_move(path, incoming / path.name)
        except Exception as exc:
            log.warning("could not move %s into Incoming: %s", path.name, exc)
            return
        log.info("queued for processing: %s", moved.name)

    # -- helpers ------------------------------------------------------------

    def _record(
        self, session: Any, source_name: str, identifier: str, file_name: Optional[str]
    ) -> Optional[Download]:
        statement = select(Download).where(
            Download.source_name == source_name, Download.identifier == identifier
        )
        if file_name:
            statement = statement.where(Download.file_name == file_name)
        row: Optional[Download] = session.execute(statement.limit(1)).scalar_one_or_none()
        return row

    def _mark_failed(
        self, source_name: str, identifier: str, file_name: Optional[str], error: str
    ) -> None:
        with self.db.write_session() as session:
            record = self._record(session, source_name, identifier, file_name)
            if record is not None:
                record.state = "failed"
                record.error = error[:4000]

    def _downloaded_bytes(self) -> int:
        with self.db.session() as session:
            return int(
                session.execute(
                    select(func.coalesce(func.sum(Download.downloaded_bytes), 0))
                ).scalar_one()
            )

    # -- introspection ------------------------------------------------------

    def statistics(self) -> dict[str, Any]:
        with self.db.session() as session:
            rows = session.execute(
                select(Download.state, func.count(), func.coalesce(func.sum(Download.size_bytes), 0))
                .group_by(Download.state)
            ).all()
        return {
            "by_state": {
                str(state): {"count": int(count), "bytes": int(size)}
                for state, count, size in rows
            },
            "sources": [s.describe() for s in all_sources()],
            "destination": str(self.destination),
        }


def _safe_name(identifier: str, file_name: str) -> str:
    """Namespace the downloaded file by its item id.

    Archive items routinely contain a file called ``video.mp4``; without the
    identifier prefix the second download would collide with the first.
    """
    from ..util.atomic import safe_filename

    stem = safe_filename(identifier, max_bytes=100)
    return f"{stem}__{safe_filename(file_name, max_bytes=100)}"
