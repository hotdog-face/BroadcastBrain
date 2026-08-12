"""Fieldstation 42 export.

Produces a self-contained directory that Fieldstation (or any other player)
can consume without ever touching BroadcastBrain's database:

    Fieldstation Export/
      metadata.json          every exported clip, fully described
      channels.json          suggested channels with inventory counts
      schedule.json          ready-made blocks with commercial breaks
      catalog/<Category>.json   per-category listings
      thumbnails/            copied stills, so the export is portable
      README.md              what the files are and how to point FS42 at them

The export is regenerated wholesale each time rather than patched. It is
cheap, it removes an entire class of drift bug, and it means a corrupted
export is fixed by running the command again.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..constants import EXPORT_SCHEMA_VERSION
from ..db.base import Database
from ..db.models import Clip
from ..db.repo import ClipRepository, EntityRepository, TagRepository
from ..errors import ExportError
from ..library.layout import category_directory, export_slug
from ..types import Category, ClipStatus
from ..util.atomic import write_json
from .scheduling import (
    ExportClip,
    SchedulingConfig,
    build_schedule,
    schedule_statistics,
    suggest_channels,
)

log = logging.getLogger(__name__)


@dataclass
class ExportSummary:
    """What an export run produced."""

    root: Path
    clip_count: int
    program_count: int
    interstitial_count: int
    channel_count: int
    block_count: int
    thumbnails_copied: int
    generated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "clips": self.clip_count,
            "programs": self.program_count,
            "interstitials": self.interstitial_count,
            "channels": self.channel_count,
            "blocks": self.block_count,
            "thumbnails": self.thumbnails_copied,
            "generated_at": self.generated_at,
        }


class FieldstationExporter:
    """Writes the export directory."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.root = settings.paths.require("export")

    def export(self, *, limit: Optional[int] = None) -> ExportSummary:
        """Regenerate the whole export."""
        if not self.settings.export.enabled:
            raise ExportError("export is disabled in configuration")

        clips, documents = self._collect(limit=limit)
        if not clips:
            log.warning("nothing to export yet: no clips are ready")

        self.root.mkdir(parents=True, exist_ok=True)
        generated_at = datetime.now(timezone.utc).isoformat()

        programs = [c for c in clips if c.is_program]
        interstitials = [c for c in clips if c.is_interstitial]

        config = SchedulingConfig(
            break_interval_s=self.settings.export.break_interval_s,
            pod_target_s=self.settings.export.pod_target_s,
            pod_min_spots=self.settings.export.pod_min_spots,
            pod_max_spots=self.settings.export.pod_max_spots,
            return_bumper_probability=self.settings.export.return_bumper_probability,
            seed=self.settings.export.random_seed,
        )
        blocks = build_schedule(programs, interstitials, config)
        channels = suggest_channels(clips)

        write_json(
            self.root / "metadata.json",
            {
                "schema": EXPORT_SCHEMA_VERSION,
                "generated_at": generated_at,
                "instance": self.settings.instance_name,
                "era": list(self.settings.era_range()),
                "counts": {
                    "clips": len(clips),
                    "programs": len(programs),
                    "interstitials": len(interstitials),
                },
                "clips": documents,
            },
        )
        write_json(
            self.root / "channels.json",
            {
                "schema": EXPORT_SCHEMA_VERSION,
                "generated_at": generated_at,
                "channels": channels,
            },
        )
        write_json(
            self.root / "schedule.json",
            {
                "schema": EXPORT_SCHEMA_VERSION,
                "generated_at": generated_at,
                "config": {
                    "break_interval_s": config.break_interval_s,
                    "pod_target_s": config.pod_target_s,
                    "seed": config.seed,
                },
                "statistics": schedule_statistics(blocks),
                "blocks": [block.as_dict() for block in blocks],
            },
        )
        self._write_catalog(clips, generated_at)
        thumbnails = self._copy_thumbnails(documents)
        self._write_readme(generated_at, len(clips), len(channels))

        summary = ExportSummary(
            root=self.root,
            clip_count=len(clips),
            program_count=len(programs),
            interstitial_count=len(interstitials),
            channel_count=len(channels),
            block_count=len(blocks),
            thumbnails_copied=thumbnails,
            generated_at=generated_at,
        )
        log.info(
            "export written to %s: %d clips, %d channels, %d blocks",
            self.root, summary.clip_count, summary.channel_count, summary.block_count,
        )
        return summary

    # -- collection ---------------------------------------------------------

    def _collect(self, *, limit: Optional[int]) -> tuple[list[ExportClip], list[dict[str, Any]]]:
        """Read every exportable clip once, building both views together."""
        clips: list[ExportClip] = []
        documents: list[dict[str, Any]] = []

        with self.db.session() as session:
            rows = ClipRepository.ready_for_export(
                session, min_confidence=self.settings.export.min_confidence, limit=limit
            )
            for clip in rows:
                if not clip.file_path or not Path(clip.file_path).exists():
                    continue
                tags = TagRepository.for_clip(session, clip.id)
                entities = EntityRepository.for_clip(session, clip.id)
                clips.append(_to_export_clip(clip))
                documents.append(_to_document(clip, tags, entities))
        return clips, documents

    # -- writers ------------------------------------------------------------

    def _write_catalog(self, clips: list[ExportClip], generated_at: str) -> None:
        """One JSON file per category -- the simplest possible consumer API."""
        catalog_dir = self.root / "catalog"
        catalog_dir.mkdir(parents=True, exist_ok=True)

        grouped: dict[Category, list[ExportClip]] = {}
        for clip in clips:
            grouped.setdefault(clip.category, []).append(clip)

        for category in Category:
            entries = grouped.get(category, [])
            write_json(
                catalog_dir / f"{category_directory(category).replace(' ', '_').lower()}.json",
                {
                    "category": category.value,
                    "directory": category_directory(category),
                    "generated_at": generated_at,
                    "count": len(entries),
                    "total_duration_s": round(sum(c.duration_s for c in entries), 1),
                    "clips": [c.as_dict() for c in entries],
                },
            )

    def _copy_thumbnails(self, documents: list[dict[str, Any]]) -> int:
        """Copy stills into the export so it can be moved to another machine."""
        target_dir = self.root / "thumbnails"
        target_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for document in documents:
            source = document.get("thumbnail")
            if not source:
                continue
            source_path = Path(str(source))
            if not source_path.exists():
                continue
            target = target_dir / f"{document['id']}.jpg"
            try:
                shutil.copy2(source_path, target)
            except OSError:  # pragma: no cover - non-fatal
                continue
            document["thumbnail_export"] = str(target.relative_to(self.root))
            copied += 1
        return copied

    def _write_readme(self, generated_at: str, clip_count: int, channel_count: int) -> None:
        text = f"""# BroadcastBrain -> Fieldstation 42 export

Generated {generated_at} by `{self.settings.instance_name}`.
{clip_count} clips, {channel_count} suggested channels.

| File | Contents |
| --- | --- |
| `metadata.json` | Every exported clip with full metadata and absolute media paths. |
| `channels.json` | Suggested channels, one per network, with inventory counts and a readiness flag. |
| `schedule.json` | Pre-built blocks: each program with its commercial pods, promos and return bumpers. |
| `catalog/*.json` | One listing per category (commercials, tv_shows, promos, ...). |
| `thumbnails/` | A still for each clip, named `<clip-id>.jpg`. |

## Using this with Fieldstation 42

`channels.json` tells you which networks have enough material to make a
channel: entries with `"ready": true` have both programming and interstitials.
Point a Fieldstation catalog at the corresponding `Library/` folders, or read
`schedule.json` directly if you want the pod structure this export already
worked out.

Media paths in `metadata.json` are absolute and point into the library. Set
`export.copy_media = true` in the BroadcastBrain config if you would rather the
export be fully self-contained.

Everything here is regenerated from scratch on each export; edit the library or
the database, not these files.
"""
        (self.root / "README.md").write_text(text, encoding="utf-8")


def _to_export_clip(clip: Clip) -> ExportClip:
    try:
        category = Category(clip.category)
    except ValueError:
        category = Category.UNKNOWN
    return ExportClip(
        id=clip.id,
        path=str(clip.file_path),
        duration_s=float(clip.duration_s),
        category=category,
        title=clip.title or "",
        network=clip.network,
        series=clip.series,
        air_year=clip.air_year,
        subcategory=clip.subcategory,
        thumbnail=clip.thumbnail_path,
    )


def _to_document(clip: Clip, tags: list[str], entities: list[tuple[Any, float]]) -> dict[str, Any]:
    """The per-clip record inside ``metadata.json``.

    Flatter than the library sidecar on purpose: a scheduler wants to filter on
    fields, not walk a nested document.
    """
    grouped: dict[str, list[str]] = {}
    for entity, confidence in entities:
        if confidence >= 0.4:
            grouped.setdefault(entity.kind, []).append(entity.name)

    return {
        "id": clip.id,
        "slug": export_slug(clip),
        "path": clip.file_path,
        "thumbnail": clip.thumbnail_path,
        "duration_s": round(clip.duration_s, 3),
        "category": clip.category,
        "subcategory": clip.subcategory,
        "confidence": round(clip.category_confidence, 3),
        "title": clip.title,
        "series": clip.series,
        "network": clip.network,
        "air_year": clip.air_year,
        "genre": clip.genre,
        "summary": clip.summary,
        "tags": tags,
        "people": grouped.get("person", []),
        "brands": grouped.get("brand", []),
        "products": grouped.get("product", []),
        "shows": grouped.get("show", []),
        "songs": grouped.get("song", []),
        "width": clip.width,
        "height": clip.height,
        "aspect_ratio": clip.aspect_ratio,
        "fps": clip.fps,
        "quality_score": clip.quality_score,
        "is_commercial": clip.is_commercial,
        "is_promo": clip.is_promo,
        "is_episode": clip.is_episode,
        "is_movie": clip.is_movie,
        "is_music_video": clip.is_music_video,
        "status": clip.status or ClipStatus.READY.value,
    }
