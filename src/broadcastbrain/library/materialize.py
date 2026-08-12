"""Moving finished clips into the library, with their metadata sidecars.

This is the last stage before a clip is visible to Fieldstation, and the one
with the strictest correctness requirement: a clip must appear in the library
complete or not at all. Every move is atomic, and the database is only updated
after the file is safely in place -- so an interruption leaves the clip
"analyzed but not materialized", which the next run simply redoes.

A JSON sidecar is written next to nothing in particular (it lives in
``Metadata/``) so the library stays browsable and, in the worst case, fully
rebuildable from disk alone if the database were ever lost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..db.base import Database
from ..db.models import Clip, Source
from ..db.repo import (
    ClipRepository,
    EntityRepository,
    EventRepository,
    TagRepository,
    TextRepository,
)
from ..errors import StorageError
from ..types import ClipStatus, Stage
from ..util.atomic import atomic_move, free_space_gb, safe_filename, write_json
from .layout import (
    clip_filename,
    clip_relative_path,
    ensure_library_tree,
    metadata_filename,
)

log = logging.getLogger(__name__)


@dataclass
class MaterializeResult:
    clip_id: str
    library_path: Optional[Path]
    metadata_path: Optional[Path]
    moved: bool
    reason: str = ""


class Materializer:
    """Places clips into their final library location."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.library_root = settings.paths.require("library")
        ensure_library_tree(self.library_root)

    def materialize(self, clip_id: str) -> MaterializeResult:
        """Move one clip into the library and write its metadata sidecar."""
        with self.db.session() as session:
            clip = ClipRepository.get(session, clip_id)
            if clip is None:
                return MaterializeResult(clip_id, None, None, False, "clip not found")
            source = session.get(Source, clip.source_id)
            document = build_metadata_document(session, clip, source)
            session.expunge(clip)

        self._check_space()

        source_path = Path(clip.processing_path) if clip.processing_path else None
        if source_path is None or not source_path.exists():
            # Already materialized (a resumed run) is a success, not an error.
            if clip.file_path and Path(clip.file_path).exists():
                return self._finish(clip, Path(clip.file_path), document, moved=False)
            return MaterializeResult(clip_id, None, None, False, "no media file to move")

        target = self._target_for(clip, source_path.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)

        if self.settings.dry_run:
            log.info("[dry-run] would move %s -> %s", source_path.name, target)
            return MaterializeResult(clip_id, target, None, False, "dry run")

        try:
            final_path = atomic_move(source_path, target, overwrite=True)
        except StorageError as exc:
            log.error("could not materialize clip %s: %s", clip_id[:8], exc)
            return MaterializeResult(clip_id, None, None, False, str(exc))

        return self._finish(clip, final_path, document, moved=True)

    # -- internals ----------------------------------------------------------

    def _target_for(self, clip: Clip, suffix: str) -> Path:
        """Where this clip belongs: the library, or the duplicates folder.

        An alternate lives in ``Duplicates/<canonical-id>/`` and *not* in the
        library. Filing it in both would mean browsing ``Commercials/`` shows
        four copies of the same spot, which defeats the point of having
        deduplicated at all. The clip row keeps it fully searchable either way.
        """
        if not clip.is_alternate:
            return self.library_root / clip_relative_path(clip, extension=suffix)

        canonical = clip.duplicate_of or "unresolved"
        return (
            self.settings.paths.require("duplicates")
            / safe_filename(canonical)
            / clip_filename(clip, extension=suffix)
        )

    def _finish(
        self, clip: Clip, final_path: Path, document: dict[str, Any], *, moved: bool
    ) -> MaterializeResult:
        metadata_path = self.settings.paths.require("metadata") / metadata_filename(clip)
        document["file_path"] = str(final_path)
        document["materialized_at"] = datetime.now().astimezone().isoformat()
        try:
            write_json(metadata_path, document)
        except OSError as exc:  # pragma: no cover - disk issue
            log.warning("could not write metadata sidecar for %s: %s", clip.id[:8], exc)
            metadata_path = None  # type: ignore[assignment]

        size = final_path.stat().st_size if final_path.exists() else None
        with self.db.write_session() as session:
            ClipRepository.update_fields(
                session,
                clip.id,
                file_path=str(final_path),
                processing_path=None,
                metadata_path=str(metadata_path) if metadata_path else None,
                size_bytes=size,
                status=ClipStatus.READY.value,
            )
            ClipRepository.set_stage(session, clip.id, Stage.MATERIALIZED)
            EventRepository.log(
                session,
                "materialized",
                f"{clip.category}: {final_path.name}",
                clip_id=clip.id,
                detail={"path": str(final_path), "moved": moved},
            )
        log.debug("clip %s -> %s", clip.id[:8], final_path)
        return MaterializeResult(clip.id, final_path, metadata_path, moved)

    def _check_space(self) -> None:
        """Refuse to fill the volume. Running out of disk mid-move is ugly."""
        free = free_space_gb(self.library_root)
        if free < self.settings.workers.min_free_gb:
            from ..errors import RetryableError

            raise RetryableError(
                f"only {free:.1f} GB free on the library volume; "
                f"minimum is {self.settings.workers.min_free_gb} GB",
                retry_after_s=600,
            )


def build_metadata_document(session: Any, clip: Clip, source: Optional[Source]) -> dict[str, Any]:
    """The complete, self-contained record of a clip.

    Written to ``Metadata/<uuid>.json`` and reused verbatim by the Fieldstation
    export. Everything the brief asked to be stored per clip is here, which is
    what makes the library reconstructible from disk alone.
    """
    texts = TextRepository.all_for_clip(session, clip.id)
    tags = TagRepository.for_clip(session, clip.id)
    entities = EntityRepository.for_clip(session, clip.id)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entity, confidence in entities:
        grouped.setdefault(entity.kind, []).append(
            {"name": entity.name, "confidence": round(confidence, 3)}
        )

    return {
        "id": clip.id,
        "schema": 1,
        "source": {
            "id": clip.source_id,
            "filename": source.filename if source else None,
            "original_path": source.original_path if source else None,
            "kind": source.kind if source else None,
            "sha256": source.sha256 if source else None,
        },
        "timing": {
            "start_s": round(clip.start_s, 3),
            "end_s": round(clip.end_s, 3),
            "duration_s": round(clip.duration_s, 3),
            "sequence": clip.seq,
        },
        "technical": {
            "width": clip.width,
            "height": clip.height,
            "aspect_ratio": clip.aspect_ratio,
            "fps": clip.fps,
            "video_codec": clip.video_codec,
            "audio_codec": clip.audio_codec,
            "video_bitrate": clip.video_bitrate,
            "audio_bitrate": clip.audio_bitrate,
            "sample_rate": clip.sample_rate,
            "channels": clip.channels,
            "interlaced": clip.interlaced,
            "size_bytes": clip.size_bytes,
            "quality_score": clip.quality_score,
            "quality_notes": clip.quality_notes,
        },
        "classification": {
            "category": clip.category,
            "subcategory": clip.subcategory,
            "confidence": clip.category_confidence,
            "evidence": clip.category_evidence,
            "is_commercial": clip.is_commercial,
            "is_promo": clip.is_promo,
            "is_episode": clip.is_episode,
            "is_movie": clip.is_movie,
            "is_music_video": clip.is_music_video,
            "has_intro": clip.has_intro,
            "has_credits": clip.has_credits,
        },
        "identification": {
            "title": clip.title,
            "series": clip.series,
            "episode_title": clip.episode_title,
            "network": clip.network,
            "network_confidence": clip.network_confidence,
            "air_year": clip.air_year,
            "air_year_confidence": clip.air_year_confidence,
            "genre": clip.genre,
            "language": clip.language,
        },
        "content": {
            "summary": clip.summary,
            "transcript": texts.get("transcript", ""),
            "ocr_text": texts.get("ocr", ""),
            "tags": tags,
            "entities": grouped,
        },
        "identity": {
            "sha256": clip.sha256,
            "chromaprint": clip.chromaprint,
            "duplicate_of": clip.duplicate_of,
            "is_alternate": clip.is_alternate,
            "duplicate_similarity": clip.duplicate_similarity,
        },
        "assets": {
            "thumbnail": clip.thumbnail_path,
            "transcript_file": clip.transcript_path,
        },
    }
