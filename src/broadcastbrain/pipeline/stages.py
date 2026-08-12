"""Stage handlers: one function per job kind.

Each handler does one thing and enqueues the next step. That chaining -- rather
than one long per-source function -- is what makes the pipeline resumable at
fine granularity: a crash costs the single stage that was in flight, not a
whole six-hour recording.

Every handler is **idempotent**. Running a stage twice on the same input
produces the same result and does no duplicate work, because each one checks
what already exists before doing anything expensive. That property is what
allows the queue to retry freely and the operator to re-run anything at will.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..analyze.runner import AnalysisRunner
from ..categorize.classifier import Categorizer
from ..config import Settings
from ..db.base import Database
from ..db.models import Job
from ..db.queue import JobQueue
from ..db.repo import ClipRepository, EventRepository, SourceRepository
from ..dedupe.hashes import compute_clip_fingerprints
from ..dedupe.resolver import Deduplicator
from ..errors import PermanentError, RetryableError, SegmentationError, UnsupportedMediaError
from ..export.fieldstation import FieldstationExporter
from ..library.materialize import Materializer
from ..media.disc import concat_input, is_disc_image, is_dvd_directory, plan_disc_expansion
from ..media.ffmpeg import FFmpeg
from ..media.probe import MediaInfo, probe
from ..search.fts import get_backend
from ..segment.segmenter import Segmenter
from ..types import (
    ClipStatus,
    JobKind,
    SourceKind,
    SourceStatus,
    Stage,
)
from ..util.hashing import sha256_file
from .discovery import mark_source_complete

log = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


@dataclass
class StageContext:
    """Shared services handed to every stage handler."""

    settings: Settings
    db: Database
    queue: JobQueue
    progress: ProgressFn

    def note(self, fraction: float, message: str) -> None:
        self.progress(fraction, message)


# ---------------------------------------------------------------------------
# 1. Ingest: probe the source, expand discs
# ---------------------------------------------------------------------------


def handle_ingest_source(ctx: StageContext, job: Job) -> None:
    """Probe a discovered source and prepare it for segmentation."""
    source_id = str(job.payload["source_id"])
    with ctx.db.session() as session:
        source = SourceRepository.get(session, source_id)
        if source is None:
            raise PermanentError(f"source {source_id} no longer exists")
        path = Path(source.path)
        kind = SourceKind(source.kind)

    if not path.exists():
        _fail_source(ctx, source_id, f"file has disappeared: {path}")
        raise PermanentError(f"source file missing: {path}")

    with ctx.db.write_session() as session:
        SourceRepository.set_status(session, source_id, SourceStatus.PROCESSING)

    # Discs become several child sources, one per title.
    if is_disc_image(path) or is_dvd_directory(path):
        _expand_disc(ctx, source_id, path)
        return

    ctx.note(0.1, f"probing {path.name}")
    try:
        info: MediaInfo = probe(path, ffprobe_binary=ctx.settings.ffprobe_binary)
    except UnsupportedMediaError as exc:
        _fail_source(ctx, source_id, str(exc), reject=True)
        raise PermanentError(str(exc)) from exc

    # The full hash is deferred to here rather than done at discovery, so a
    # scan of a large Incoming folder stays fast.
    ctx.note(0.4, f"hashing {path.name}")
    digest = sha256_file(path)

    with ctx.db.write_session() as session:
        duplicate = SourceRepository.by_sha256(session, digest)
        if duplicate is not None and duplicate.id != source_id:
            SourceRepository.set_status(
                session,
                source_id,
                SourceStatus.REJECTED,
                reject_reason=f"identical to source {duplicate.id}",
            )
            EventRepository.log(
                session,
                "duplicate_source",
                f"{path.name} is byte-identical to {duplicate.filename}",
                source_id=source_id,
            )
            log.info("source %s is a duplicate of %s; skipping", path.name, duplicate.filename)
            return

        source = SourceRepository.get(session, source_id)
        if source is not None:
            source.sha256 = digest
            source.media_info = info.to_dict()
            source.duration_s = info.duration_s
            source.container = info.container
            source.width = info.width
            source.height = info.height
            source.fps = info.fps
            if kind is SourceKind.UNKNOWN:
                source.kind = info.guess_source_kind().value
            source.stage = Stage.PROBED.value

    log.info(
        "%s: %s, %.1f min, %sx%s",
        path.name, info.container, info.duration_s / 60.0, info.width, info.height,
    )
    ctx.queue.enqueue(JobKind.SEGMENT_SOURCE, {"source_id": source_id})


def _expand_disc(ctx: StageContext, source_id: str, path: Path) -> None:
    """Turn a disc image or rip into one child source per title."""
    work_dir = ctx.settings.paths.require("processing") / "discs"
    work_dir.mkdir(parents=True, exist_ok=True)

    titles = plan_disc_expansion(path, work_dir)
    created = 0
    for title in titles:
        input_spec = concat_input(title)
        with ctx.db.write_session() as session:
            existing = SourceRepository.by_path(session, Path(input_spec))
            if existing is not None:
                continue
            child = SourceRepository.create(
                session,
                path=Path(input_spec),
                size_bytes=title.size_bytes,
                kind=SourceKind.DVD,
                origin="disc",
                origin_ref=f"{path.name}#title{title.number}",
                parent_source_id=source_id,
            )
            child_id = child.id
            child.filename = f"{path.stem}_title{title.number:02d}"
        ctx.queue.enqueue(JobKind.INGEST_SOURCE, {"source_id": child_id})
        created += 1

    with ctx.db.write_session() as session:
        source = SourceRepository.get(session, source_id)
        if source is not None:
            source.stage = Stage.SEGMENTED.value
            source.clip_count = 0
            source.segmentation = {"disc_titles": len(titles), "children_created": created}
        SourceRepository.set_status(session, source_id, SourceStatus.COMPLETE)
        EventRepository.log(
            session,
            "disc_expanded",
            f"{path.name}: {len(titles)} title(s), {created} queued",
            source_id=source_id,
        )
    log.info("%s expanded into %d title(s)", path.name, created)


# ---------------------------------------------------------------------------
# 2. Segment: signals -> boundaries -> clips on disk
# ---------------------------------------------------------------------------


def handle_segment_source(ctx: StageContext, job: Job) -> None:
    """Split a source into clips and cut them into ``Processing/``."""
    source_id = str(job.payload["source_id"])
    with ctx.db.session() as session:
        source = SourceRepository.get(session, source_id)
        if source is None:
            raise PermanentError(f"source {source_id} no longer exists")
        path = Path(source.path)
        kind = SourceKind(source.kind)
        probed = source.media_info is not None

    if not probed:
        # The ingest stage owns probing; a segment job that arrives first is
        # retried rather than guessing at the source's properties.
        raise RetryableError("source has not been probed yet")
    if not path.exists():
        _fail_source(ctx, source_id, f"file has disappeared: {path}")
        raise PermanentError(f"source file missing: {path}")

    media = probe(path, ffprobe_binary=ctx.settings.ffprobe_binary)

    ctx.note(0.05, f"analyzing signals in {path.name}")
    segmenter = Segmenter(ctx.settings)
    try:
        result = segmenter.segment(
            path, media, source_kind=kind, progress=lambda f, m: ctx.note(0.05 + f * 0.45, m)
        )
    except SegmentationError as exc:
        _fail_source(ctx, source_id, str(exc))
        raise PermanentError(str(exc)) from exc

    with ctx.db.write_session() as session:
        clips = ClipRepository.upsert_segments(session, source_id, result.segments)
        clip_ids = [clip.id for clip in clips]
        source = SourceRepository.get(session, source_id)
        if source is not None:
            source.clip_count = len(clips)
            source.segmentation = result.as_dict()
            source.stage = Stage.SEGMENTED.value
        EventRepository.log(
            session,
            "segmented",
            f"{path.name}: {len(clips)} clip(s), confidence {result.confidence:.2f}",
            source_id=source_id,
            detail={"strategy": result.strategy, "confidence": result.confidence},
        )

    _cut_clips(ctx, source_id, path, clip_ids)

    for clip_id in clip_ids:
        ctx.queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": clip_id})
    log.info("%s: %d clip(s) queued for analysis", path.name, len(clip_ids))


def _cut_clips(ctx: StageContext, source_id: str, source_path: Path, clip_ids: list[str]) -> None:
    """Cut each segment out of the source, skipping any already on disk."""
    ffmpeg = FFmpeg(ctx.settings)
    output_dir = ctx.settings.paths.require("processing") / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(clip_ids)

    for index, clip_id in enumerate(clip_ids):
        with ctx.db.session() as session:
            clip = ClipRepository.get(session, clip_id)
            if clip is None:
                continue
            start, end, seq = clip.start_s, clip.end_s, clip.seq
            existing = clip.processing_path or clip.file_path

        if existing and Path(existing).exists():
            continue  # already cut on a previous run

        target = output_dir / f"{seq:05d}_{clip_id[:8]}{_container_suffix(source_path)}"
        ctx.note(0.5 + 0.5 * (index / max(1, total)), f"cutting clip {index + 1}/{total}")

        if ctx.settings.dry_run:
            continue
        try:
            cut = ffmpeg.cut(source_path, target, start_s=start, end_s=end)
        except Exception as exc:
            log.warning("could not cut clip %s: %s", clip_id[:8], exc)
            with ctx.db.write_session() as session:
                ClipRepository.set_status(session, clip_id, ClipStatus.FAILED, error=str(exc))
            continue

        with ctx.db.write_session() as session:
            ClipRepository.update_fields(
                session,
                clip_id,
                processing_path=str(cut.path),
                size_bytes=cut.path.stat().st_size if cut.path.exists() else None,
                status=ClipStatus.PROCESSING.value,
            )
            ClipRepository.set_stage(session, clip_id, Stage.EXTRACTED)


def _container_suffix(source_path: Path) -> str:
    """Choose an output container that can hold the source's streams.

    Matroska is the safe default: it accepts essentially any codec, which
    matters when stream-copying MPEG-2 out of a VOB or DV out of a VHS capture.
    """
    suffix = source_path.suffix.lower()
    if suffix in {".mp4", ".m4v", ".mkv", ".webm"}:
        return suffix
    return ".mkv"


# ---------------------------------------------------------------------------
# 3. Analyze
# ---------------------------------------------------------------------------


def handle_analyze_clip(ctx: StageContext, job: Job) -> None:
    """Run the analyzer suite over one clip."""
    clip_id = str(job.payload["clip_id"])
    force = job.payload.get("force")
    runner = AnalysisRunner(ctx.settings, ctx.db)

    outcome = runner.analyze(
        clip_id,
        force=list(force) if isinstance(force, list) else None,
        progress=ctx.progress,
    )

    with ctx.db.write_session() as session:
        ClipRepository.set_stage(session, clip_id, Stage.ANALYZED)
        if outcome.failed:
            EventRepository.log(
                session,
                "analysis_partial",
                f"{len(outcome.failed)} analyzer(s) failed",
                level="warning",
                clip_id=clip_id,
                detail=outcome.as_dict(),
            )

    _enqueue_next(ctx, JobKind.FINGERPRINT_CLIP, clip_id)


# ---------------------------------------------------------------------------
# 4. Fingerprint
# ---------------------------------------------------------------------------


def handle_fingerprint_clip(ctx: StageContext, job: Job) -> None:
    """Compute the identity hashes used for deduplication."""
    clip_id = str(job.payload["clip_id"])
    with ctx.db.session() as session:
        clip = ClipRepository.get(session, clip_id)
        if clip is None:
            raise PermanentError(f"clip {clip_id} no longer exists")
        media_path = clip.processing_path or clip.file_path
        duration = clip.duration_s
        has_hashes = bool(clip.sha256 and clip.phash)

    if not media_path or not Path(media_path).exists():
        log.warning("clip %s has no media file to fingerprint", clip_id[:8])
        _enqueue_next(ctx, JobKind.DEDUPE_CLIP, clip_id)
        return

    if not has_hashes:
        ctx.note(0.2, "fingerprinting")
        fingerprints = compute_clip_fingerprints(
            ctx.settings,
            Path(media_path),
            duration,
            frame_count=ctx.settings.dedupe.phash_frames,
        )
        with ctx.db.write_session() as session:
            ClipRepository.update_fields(
                session,
                clip_id,
                sha256=str(fingerprints["sha256"]),
                phash=fingerprints["phash"],
                phash_head=fingerprints["phash_head"],
            )

    with ctx.db.write_session() as session:
        ClipRepository.set_stage(session, clip_id, Stage.FINGERPRINTED)
    _enqueue_next(ctx, JobKind.DEDUPE_CLIP, clip_id)


# ---------------------------------------------------------------------------
# 5. Deduplicate
# ---------------------------------------------------------------------------


def handle_dedupe_clip(ctx: StageContext, job: Job) -> None:
    """Find and resolve duplicates of this clip."""
    clip_id = str(job.payload["clip_id"])
    deduplicator = Deduplicator(ctx.settings, ctx.db)
    outcome = deduplicator.process(clip_id)

    if outcome.found_duplicates:
        log.info(
            "clip %s: %d duplicate(s); canonical is %s",
            clip_id[:8], len(outcome.matches), (outcome.canonical_id or "?")[:8],
        )

    with ctx.db.write_session() as session:
        ClipRepository.set_stage(session, clip_id, Stage.DEDUPED)
    _enqueue_next(ctx, JobKind.CATEGORIZE_CLIP, clip_id)


# ---------------------------------------------------------------------------
# 6. Categorize
# ---------------------------------------------------------------------------


def handle_categorize_clip(ctx: StageContext, job: Job) -> None:
    """Decide what the clip is, from the votes analyzers already stored."""
    clip_id = str(job.payload["clip_id"])
    categorizer = Categorizer(ctx.settings, ctx.db)
    decision = categorizer.categorize(clip_id)
    if decision is None:
        raise PermanentError(f"clip {clip_id} no longer exists")

    _enqueue_next(ctx, JobKind.MATERIALIZE_CLIP, clip_id)


# ---------------------------------------------------------------------------
# 7. Materialize
# ---------------------------------------------------------------------------


def handle_materialize_clip(ctx: StageContext, job: Job) -> None:
    """Move the clip into the library and index it for search."""
    clip_id = str(job.payload["clip_id"])
    materializer = Materializer(ctx.settings, ctx.db)
    result = materializer.materialize(clip_id)

    if result.library_path is None and result.reason:
        log.warning("clip %s not materialized: %s", clip_id[:8], result.reason)

    # Index after the file is in place, so a search hit always points at a
    # file that exists.
    backend = get_backend(ctx.db)
    with ctx.db.write_session() as session:
        backend.index_clip(session, clip_id)

    _maybe_finish_source(ctx, clip_id)


def _maybe_finish_source(ctx: StageContext, clip_id: str) -> None:
    """Mark the parent source complete once all of its clips are done."""
    with ctx.db.session() as session:
        clip = ClipRepository.get(session, clip_id)
        if clip is None:
            return
        source_id = clip.source_id
        siblings = ClipRepository.by_source(session, source_id)
        outstanding = [
            c
            for c in siblings
            if c.stage not in (Stage.MATERIALIZED.value, Stage.EXPORTED.value)
            and c.status != ClipStatus.FAILED.value
        ]
    if outstanding:
        return
    mark_source_complete(ctx.db, ctx.settings, source_id)


# ---------------------------------------------------------------------------
# 8. Export
# ---------------------------------------------------------------------------


def handle_export_library(ctx: StageContext, job: Job) -> None:  # noqa: ARG001
    """Regenerate the Fieldstation export.

    Takes ``job`` for signature uniformity with every other handler; the
    export always covers the whole library, so there is nothing to read.
    """
    exporter = FieldstationExporter(ctx.settings, ctx.db)
    summary = exporter.export()
    with ctx.db.write_session() as session:
        EventRepository.log(
            session,
            "exported",
            f"{summary.clip_count} clips, {summary.channel_count} channels",
            detail=summary.as_dict(),
        )


# ---------------------------------------------------------------------------
# 9. Download (archive sources)
# ---------------------------------------------------------------------------


def handle_download_item(ctx: StageContext, job: Job) -> None:
    """Fetch one item from an external archive source."""
    from ..sources.manager import DownloadManager

    manager = DownloadManager(ctx.settings, ctx.db, ctx.queue)
    manager.download_one(
        source_name=str(job.payload["source_name"]),
        identifier=str(job.payload["identifier"]),
        file_name=job.payload.get("file_name"),
        progress=ctx.progress,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Job kind -> handler. The worker looks up here and nowhere else.
HANDLERS: dict[JobKind, Callable[[StageContext, Job], None]] = {
    JobKind.INGEST_SOURCE: handle_ingest_source,
    JobKind.SEGMENT_SOURCE: handle_segment_source,
    JobKind.ANALYZE_CLIP: handle_analyze_clip,
    JobKind.FINGERPRINT_CLIP: handle_fingerprint_clip,
    JobKind.DEDUPE_CLIP: handle_dedupe_clip,
    JobKind.CATEGORIZE_CLIP: handle_categorize_clip,
    JobKind.MATERIALIZE_CLIP: handle_materialize_clip,
    JobKind.EXPORT_LIBRARY: handle_export_library,
    JobKind.DOWNLOAD_ITEM: handle_download_item,
}


def _enqueue_next(ctx: StageContext, kind: JobKind, clip_id: str) -> None:
    """Hand a clip to the next stage, re-running it if it already ran.

    ``requeue_if_done`` is essential rather than incidental. The stage that
    calls this has just produced new data for the clip, so whatever the next
    stage concluded last time is now stale. Without it, re-analysis updates the
    analyzer results and then stops dead -- the clip keeps its old category,
    its old duplicate status and its old library location, and the operator is
    left looking at output that silently ignores the work they just asked for.
    """
    ctx.queue.enqueue(kind, {"clip_id": clip_id}, requeue_if_done=True)


def get_handler(kind: str) -> Optional[Callable[[StageContext, Job], None]]:
    try:
        return HANDLERS[JobKind(kind)]
    except (ValueError, KeyError):
        return None


def _fail_source(
    ctx: StageContext, source_id: str, message: str, *, reject: bool = False
) -> None:
    status = SourceStatus.REJECTED if reject else SourceStatus.FAILED
    with ctx.db.write_session() as session:
        SourceRepository.set_status(
            session,
            source_id,
            status,
            error=message,
            reject_reason=message if reject else None,
        )
        EventRepository.log(
            session, "source_failed", message, level="error", source_id=source_id
        )
    log.error("source %s: %s", source_id[:8], message)
