"""Finding new material in ``Incoming/``.

Two problems have to be solved here, and both are about not doing work twice:

* **Stability.** A file being copied over the network, or still being written
  by a capture card, must not be ingested mid-write. Discovery waits until a
  file's size and mtime have been unchanged for a configured interval.
* **Identity.** A file that has already been processed must be recognized
  before anything expensive happens. That is a two-step check: a cheap
  size+head+tail signature first, and only if that hits do we pay for a full
  SHA256 to confirm. Re-dropping a 40 GB capture costs a few seconds, not a
  night.

Unsupported files are moved to ``Rejected/`` with a reason recorded. Nothing is
ever deleted.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import update

from ..config import Settings
from ..constants import (
    DISC_IMAGE_EXTENSIONS,
    FILE_STABILITY_SECONDS,
    MIN_SOURCE_BYTES,
    VIDEO_EXTENSIONS,
)
from ..db.base import Database
from ..db.models import Source, utcnow
from ..db.queue import JobQueue
from ..db.repo import EventRepository, SourceRepository
from ..media.disc import is_dvd_directory
from ..types import JobKind, SourceKind, SourceStatus
from ..util.atomic import atomic_move
from ..util.hashing import quick_signature, sha256_file

log = logging.getLogger(__name__)


@dataclass
class DiscoveryResult:
    scanned: int = 0
    new_sources: int = 0
    already_known: int = 0
    unstable: int = 0
    rejected: int = 0
    errors: int = 0
    new_source_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "scanned": self.scanned,
            "new_sources": self.new_sources,
            "already_known": self.already_known,
            "unstable": self.unstable,
            "rejected": self.rejected,
            "errors": self.errors,
        }

    def summary(self) -> str:
        return (
            f"{self.scanned} file(s) seen: {self.new_sources} new, "
            f"{self.already_known} known, {self.unstable} still settling, "
            f"{self.rejected} rejected"
        )


class Discovery:
    """Scans ``Incoming/`` and registers new sources."""

    def __init__(self, settings: Settings, db: Database, queue: JobQueue) -> None:
        self.settings = settings
        self.db = db
        self.queue = queue
        self.incoming = settings.paths.require("incoming")
        self.rejected = settings.paths.require("rejected")

    def scan(self, *, stability_s: float = FILE_STABILITY_SECONDS) -> DiscoveryResult:
        """One pass over Incoming. Safe to call repeatedly and concurrently."""
        result = DiscoveryResult()
        self.incoming.mkdir(parents=True, exist_ok=True)

        for path in self._candidates():
            result.scanned += 1
            try:
                self._process(path, result, stability_s)
            except Exception as exc:
                result.errors += 1
                log.exception("discovery failed for %s", path)
                with self.db.write_session() as session:
                    EventRepository.log(
                        session, "discovery_error", f"{path.name}: {exc}", level="error"
                    )

        if result.new_sources or result.rejected:
            log.info("discovery: %s", result.summary())
        return result

    # -- candidate enumeration ---------------------------------------------

    def _candidates(self) -> Iterator[Path]:
        """Yield files and DVD directories worth considering.

        Directories that look like DVD rips are yielded whole; their contents
        are not yielded individually, because a ``VIDEO_TS`` folder is one
        source, not forty VOB files.
        """
        for entry in sorted(self.incoming.rglob("*")):
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if is_dvd_directory(entry):
                    yield entry
                continue
            if _inside_dvd_directory(entry, self.incoming):
                continue
            if entry.is_file():
                yield entry

    # -- per-file handling --------------------------------------------------

    def _process(self, path: Path, result: DiscoveryResult, stability_s: float) -> None:
        if path.is_dir():
            self._process_disc_directory(path, result)
            return

        suffix = path.suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            self._reject(path, f"unsupported extension '{suffix or 'none'}'", result)
            return

        try:
            stat = path.stat()
        except OSError as exc:
            result.errors += 1
            log.warning("cannot stat %s: %s", path, exc)
            return

        if stat.st_size < MIN_SOURCE_BYTES:
            self._reject(path, f"file is only {stat.st_size} bytes", result)
            return

        if not self._is_stable(path, stability_s):
            result.unstable += 1
            log.debug("%s is still being written; will retry next scan", path.name)
            return

        signature = quick_signature(path)
        if self._already_known(path, signature):
            result.already_known += 1
            return

        source_id = self._register(path, stat.st_size, signature, _mtime(stat.st_mtime))
        if source_id:
            result.new_sources += 1
            result.new_source_ids.append(source_id)

    def _process_disc_directory(self, path: Path, result: DiscoveryResult) -> None:
        """Register a DVD rip folder as one source; titles are split later."""
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        if total < MIN_SOURCE_BYTES:
            self._reject(path, "DVD directory contains no usable data", result)
            return

        with self.db.session() as session:
            if SourceRepository.by_path(session, path) is not None:
                result.already_known += 1
                return

        with self.db.write_session() as session:
            source = SourceRepository.create(
                session,
                path=path,
                size_bytes=total,
                kind=SourceKind.DVD,
                origin="incoming",
            )
            source_id = source.id
            EventRepository.log(
                session, "discovered", f"DVD rip: {path.name}", source_id=source_id
            )
        self.queue.enqueue(JobKind.INGEST_SOURCE, {"source_id": source_id})
        result.new_sources += 1
        result.new_source_ids.append(source_id)

    # -- helpers ------------------------------------------------------------

    def _is_stable(self, path: Path, stability_s: float) -> bool:
        """True when the file has stopped changing.

        Compares two observations separated by a short sleep, and also treats a
        file whose mtime is already older than the window as settled -- so a
        rescan of a long-idle Incoming folder is not slowed down by sleeping
        once per file.
        """
        try:
            first = path.stat()
        except OSError:
            return False

        age = time.time() - first.st_mtime
        if age >= stability_s:
            return True

        time.sleep(min(2.0, stability_s / 4.0))
        try:
            second = path.stat()
        except OSError:
            return False
        return first.st_size == second.st_size and first.st_mtime == second.st_mtime

    def _already_known(self, path: Path, signature: str) -> bool:
        """Cheap identity check first, expensive confirmation only if needed."""
        with self.db.session() as session:
            if SourceRepository.by_path(session, path) is not None:
                return True
            candidates = SourceRepository.by_quick_signature(session, signature)

        if not candidates:
            return False

        # The quick signature matched something: confirm with a real hash.
        digest = sha256_file(path)
        with self.db.session() as session:
            existing = SourceRepository.by_sha256(session, digest)
        if existing is None:
            return False

        log.info(
            "%s is a duplicate of already-ingested source %s; leaving it in place",
            path.name, existing.filename,
        )
        with self.db.write_session() as session:
            EventRepository.log(
                session,
                "duplicate_source",
                f"{path.name} matches {existing.filename} (sha256)",
                source_id=existing.id,
            )
        return True

    def _register(
        self, path: Path, size: int, signature: str, mtime: Optional[datetime]
    ) -> Optional[str]:
        """Create the source row and queue its first job."""
        kind = SourceKind.DVD if path.suffix.lower() in DISC_IMAGE_EXTENSIONS else SourceKind.UNKNOWN
        with self.db.write_session() as session:
            source = SourceRepository.create(
                session,
                path=path,
                size_bytes=size,
                mtime=mtime,
                quick_signature=signature,
                kind=kind,
                origin="incoming",
            )
            source_id = source.id
            EventRepository.log(
                session,
                "discovered",
                f"{path.name} ({size / (1024**3):.2f} GB)",
                source_id=source_id,
            )
        self.queue.enqueue(JobKind.INGEST_SOURCE, {"source_id": source_id})
        log.info("discovered %s (%.2f GB)", path.name, size / (1024**3))
        return source_id

    def _reject(self, path: Path, reason: str, result: DiscoveryResult) -> None:
        """Move a file we cannot use into ``Rejected/``, with a reason.

        Rejection is a filing decision, never a deletion: the operator can look
        at the folder, see why, and act.
        """
        result.rejected += 1
        if self.settings.dry_run:
            log.info("[dry-run] would reject %s: %s", path.name, reason)
            return

        target = self.rejected / path.name
        try:
            moved = atomic_move(path, target)
        except Exception as exc:
            log.warning("could not move rejected file %s: %s", path, exc)
            return

        note = moved.with_suffix(moved.suffix + ".reason.txt")
        with contextlib.suppress(OSError):
            note.write_text(f"{reason}\noriginal path: {path}\n", encoding="utf-8")

        log.info("rejected %s: %s", path.name, reason)
        with self.db.write_session() as session:
            EventRepository.log(
                session, "rejected", f"{path.name}: {reason}", level="warning"
            )


def _inside_dvd_directory(path: Path, root: Path) -> bool:
    """True when ``path`` lives under a VIDEO_TS/BDMV folder."""
    try:
        relative = path.relative_to(root)
    except ValueError:  # pragma: no cover - path outside the root
        return False
    return any(part.upper() in {"VIDEO_TS", "BDMV", "AUDIO_TS"} for part in relative.parts)


def _mtime(timestamp: float) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):  # pragma: no cover - odd filesystems
        return None


def mark_source_complete(db: Database, settings: Settings, source_id: str) -> None:
    """Finish a source: mark it complete and file the original away.

    The original recording is moved to ``Processing/completed/`` rather than
    deleted. Re-cutting a source with better settings later is a normal thing
    to want, and it is impossible if the master was thrown away.

    **Exactly once**, even under concurrency. This is called by every worker
    that materializes a clip, so on a source whose last few clips finish within
    milliseconds of each other, several workers reach here at the same moment.
    The status transition is therefore a conditional UPDATE and only the worker
    that wins it does the move -- otherwise the second worker re-moves the file
    onto itself and the archive quietly grows a ``-2`` suffix.
    """
    now = utcnow()
    with db.write_session() as session:
        source = SourceRepository.get(session, source_id)
        if source is None:
            return
        claimed = session.execute(
            update(Source)
            .where(Source.id == source_id, Source.status != SourceStatus.COMPLETE.value)
            .values(status=SourceStatus.COMPLETE.value, completed_at=now)
        )
        if (getattr(claimed, "rowcount", 0) or 0) != 1:
            return  # another worker already completed this source
        original = Path(source.path)
        filename = source.filename

    if not settings.move_completed_sources or settings.dry_run:
        return
    if not original.exists() or original.is_dir():
        return

    completed_dir = settings.paths.require("processing") / "completed"
    if _is_within(original, completed_dir):
        return  # already archived on a previous run

    destination = completed_dir / filename
    try:
        moved = atomic_move(original, destination)
    except Exception as exc:
        log.warning("could not archive completed source %s: %s", filename, exc)
        return

    with db.write_session() as session:
        source = SourceRepository.get(session, source_id)
        if source is not None:
            source.path = str(moved)
    log.info("source complete: %s -> %s", filename, moved.parent)


def _is_within(path: Path, directory: Path) -> bool:
    """True when ``path`` already lives under ``directory``."""
    try:
        path.resolve().relative_to(directory.resolve())
    except (ValueError, OSError):
        return False
    return True
