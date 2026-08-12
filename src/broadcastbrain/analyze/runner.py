"""The analysis runner: caching, shared inputs, and persistence.

Responsibilities, in order:

1. Work out which analyzers actually need to run for this clip, by comparing
   the registry's versions against what is already stored. **This is where the
   "never reprocess" guarantee is enforced.**
2. Extract the shared inputs (frames, analysis audio) once -- and only if some
   analyzer that is going to run needs them.
3. Run each analyzer in dependency order, feeding earlier payloads to later
   ones, isolating failures to the analyzer that caused them.
4. Persist everything: analyzer rows, texts, tags, entities, embeddings, clip
   columns, transcript sidecars and thumbnails.

Nothing here knows what an analyzer *does*, which is what allows analyzers to
be added without touching the pipeline.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..config import Settings
from ..db.base import Database
from ..db.models import Clip, Source
from ..db.repo import (
    AnalysisRepository,
    ClipRepository,
    EmbeddingRepository,
    EntityRepository,
    TagRepository,
    TextRepository,
)
from ..errors import AnalyzerError, BackendUnavailableError
from ..media.ffmpeg import FFmpeg
from ..types import AnalysisStatus, Category, Evidence
from ..util.atomic import atomic_write
from .base import AnalysisContext, Analyzer, AnalyzerResult, ClipView
from .registry import enabled_analyzers

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

#: Key under which an analyzer's classification votes are stored inside its
#: payload. Recategorizing replays these instead of re-running any model.
VOTES_KEY = "__votes__"


@dataclass
class AnalysisOutcome:
    """What happened to one clip in one analysis pass."""

    clip_id: str
    ran: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        """True when nothing hard-failed. Skips are an expected outcome."""
        return not self.failed

    def as_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "ran": self.ran,
            "cached": self.cached,
            "skipped": self.skipped,
            "failed": self.failed,
            "votes": [e.as_dict() for e in self.evidence],
            "duration_s": round(self.duration_s, 2),
        }


class AnalysisRunner:
    """Runs the analyzer suite over one clip at a time."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.ffmpeg = FFmpeg(settings)

    # -- public API ---------------------------------------------------------

    def analyze(
        self,
        clip_id: str,
        *,
        force: Optional[Sequence[str]] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> AnalysisOutcome:
        """Analyze one clip, running only what is missing or out of date.

        Args:
            force: Analyzer names to re-run even if a current result exists.
                ``["*"]`` forces everything.
        """
        started = time.monotonic()
        clip_view, stored = self._load(clip_id)
        outcome = AnalysisOutcome(clip_id=clip_id)

        if clip_view is None:
            outcome.failed["__clip__"] = "clip row or media file is missing"
            return outcome

        analyzers = enabled_analyzers(self.settings.analysis)
        force_set = set(force or ())
        force_all = "*" in force_set

        pending: list[Analyzer] = []
        previous: dict[str, dict[str, Any]] = {}
        for analyzer in analyzers:
            record = stored.get(analyzer.name)
            is_current = (
                record is not None
                and record.analyzer_version == analyzer.version
                and record.status == AnalysisStatus.OK.value
            )
            if is_current and not force_all and analyzer.name not in force_set:
                outcome.cached.append(analyzer.name)
                if record is not None and isinstance(record.payload, dict):
                    previous[analyzer.name] = record.payload
                continue
            pending.append(analyzer)

        if not pending:
            log.debug("clip %s: fully cached (%d analyzers)", clip_id[:8], len(outcome.cached))
            outcome.duration_s = time.monotonic() - started
            return outcome

        workdir = self._workdir(clip_id)
        try:
            ctx = self._build_context(clip_view, workdir, pending, previous)
            self._run_all(ctx, pending, outcome, progress)
        finally:
            if not self.settings.analysis.keep_intermediates:
                shutil.rmtree(workdir, ignore_errors=True)

        outcome.duration_s = time.monotonic() - started
        log.info(
            "clip %s analyzed in %.1fs: %d ran, %d cached, %d skipped, %d failed",
            clip_id[:8], outcome.duration_s, len(outcome.ran), len(outcome.cached),
            len(outcome.skipped), len(outcome.failed),
        )
        return outcome

    # -- loading ------------------------------------------------------------

    def _load(self, clip_id: str) -> tuple[Optional[ClipView], dict[str, Any]]:
        """Fetch the clip as a detached value object plus its stored results."""
        with self.db.session() as session:
            clip: Optional[Clip] = ClipRepository.get(session, clip_id)
            if clip is None:
                return None, {}
            media_path = clip.processing_path or clip.file_path
            if not media_path or not Path(media_path).exists():
                log.warning("clip %s has no readable media file", clip_id[:8])
                return None, {}
            source: Optional[Source] = session.get(Source, clip.source_id)
            view = ClipView(
                id=clip.id,
                source_id=clip.source_id,
                path=Path(media_path),
                start_s=clip.start_s,
                end_s=clip.end_s,
                duration_s=clip.duration_s,
                seq=clip.seq,
                source_filename=source.filename if source else "",
                source_path=source.original_path if source else "",
                source_kind=source.kind if source else "unknown",
            )
            stored = AnalysisRepository.for_clip(session, clip_id)
            # Detach: analyzers must not hold a session across model inference.
            for record in stored.values():
                session.expunge(record)
            return view, stored

    def _workdir(self, clip_id: str) -> Path:
        workdir = self.settings.paths.require("cache") / "analysis" / clip_id
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    def _build_context(
        self,
        clip: ClipView,
        workdir: Path,
        pending: Sequence[Analyzer],
        previous: dict[str, dict[str, Any]],
    ) -> AnalysisContext:
        """Extract the inputs the pending analyzers actually asked for."""
        ctx = AnalysisContext(
            clip=clip,
            settings=self.settings,
            ffmpeg=self.ffmpeg,
            workdir=workdir,
            previous=dict(previous),
        )

        wants_frames = any(a.needs_frames for a in pending)
        wants_audio = any(a.needs_audio for a in pending)

        if wants_frames:
            count = self.settings.analysis.frames_per_clip
            if self.settings.workers.low_memory:
                count = max(4, count // 2)
            ctx.frames = self.ffmpeg.extract_frames(
                clip.path, workdir / "frames", count=count, duration_s=clip.duration_s
            )
            log.debug("clip %s: %d frames extracted", clip.id[:8], len(ctx.frames))

        if wants_audio:
            ctx.audio_path = self.ffmpeg.extract_audio(clip.path, workdir / "audio.wav")
            if ctx.audio_path is None:
                log.debug("clip %s: no audio track to analyze", clip.id[:8])

        return ctx

    # -- execution ----------------------------------------------------------

    def _run_all(
        self,
        ctx: AnalysisContext,
        pending: Sequence[Analyzer],
        outcome: AnalysisOutcome,
        progress: Optional[ProgressCallback],
    ) -> None:
        total = len(pending)
        for index, analyzer in enumerate(pending):
            if progress is not None:
                progress(index / total, f"analyzing: {analyzer.name}")

            if not analyzer.applicable(ctx):
                self._record(
                    ctx.clip.id, analyzer, AnalysisStatus.SKIPPED,
                    error="not applicable to this clip", duration_ms=0,
                )
                outcome.skipped[analyzer.name] = "not applicable"
                continue

            started = time.monotonic()
            try:
                result = analyzer.run(ctx)
            except BackendUnavailableError as exc:
                # The defining behaviour: a missing backend is deferred work,
                # not lost work.
                self._record(
                    ctx.clip.id, analyzer, AnalysisStatus.SKIPPED,
                    error=str(exc), duration_ms=_ms(started),
                )
                outcome.skipped[analyzer.name] = str(exc)
                continue
            except AnalyzerError as exc:
                self._record(
                    ctx.clip.id, analyzer, AnalysisStatus.FAILED,
                    error=str(exc), duration_ms=_ms(started),
                )
                outcome.failed[analyzer.name] = str(exc)
                log.warning("analyzer %s failed on clip %s: %s", analyzer.name, ctx.clip.id[:8], exc)
                continue
            except Exception as exc:
                self._record(
                    ctx.clip.id, analyzer, AnalysisStatus.FAILED,
                    error=f"unexpected: {exc}", duration_ms=_ms(started),
                )
                outcome.failed[analyzer.name] = f"unexpected: {exc}"
                log.exception("analyzer %s raised on clip %s", analyzer.name, ctx.clip.id[:8])
                continue

            self._persist(ctx, analyzer, result, _ms(started))
            ctx.previous[analyzer.name] = result.payload
            outcome.ran.append(analyzer.name)
            for category, weight, reason in result.category_votes:
                outcome.evidence.append(
                    Evidence(category=category, weight=weight, reason=reason, source=analyzer.name)
                )

        if progress is not None:
            progress(1.0, "analysis complete")

    # -- persistence --------------------------------------------------------

    def _record(
        self,
        clip_id: str,
        analyzer: Analyzer,
        status: AnalysisStatus,
        *,
        error: Optional[str] = None,
        duration_ms: int = 0,
        payload: Optional[dict[str, Any]] = None,
        confidence: Optional[float] = None,
        backend: Optional[str] = None,
    ) -> None:
        with self.db.write_session() as session:
            AnalysisRepository.upsert(
                session,
                clip_id,
                analyzer=analyzer.name,
                analyzer_version=analyzer.version,
                status=status,
                payload=payload,
                confidence=confidence,
                backend=backend,
                error=error,
                duration_ms=duration_ms,
            )

    def _persist(
        self, ctx: AnalysisContext, analyzer: Analyzer, result: AnalyzerResult, duration_ms: int
    ) -> None:
        """Write one analyzer's output in a single transaction."""
        clip_id = ctx.clip.id
        embedding = result.fields.pop("__embedding__", None)
        thumbnail = result.fields.pop("thumbnail_path", None)

        # Votes are stored inside the analyzer's payload so that re-running
        # categorization later is a pure database operation -- no model ever
        # has to run again just because a classification rule changed.
        payload = dict(result.payload)
        if result.category_votes:
            payload[VOTES_KEY] = [
                [category.value, round(weight, 4), reason]
                for category, weight, reason in result.category_votes
            ]

        with self.db.write_session() as session:
            AnalysisRepository.upsert(
                session,
                clip_id,
                analyzer=analyzer.name,
                analyzer_version=analyzer.version,
                status=AnalysisStatus.OK,
                payload=payload,
                confidence=result.confidence,
                backend=result.backend,
                duration_ms=duration_ms,
            )

            for kind, (content, detail) in result.texts.items():
                TextRepository.upsert(
                    session, clip_id, kind=kind, content=content, detail=detail
                )

            if result.tags:
                TagRepository.set_tags(session, clip_id, result.tags, source=analyzer.name)

            threshold = self.settings.analysis.min_entity_confidence
            for mention in result.entities:
                if mention.confidence >= threshold:
                    EntityRepository.link(
                        session,
                        clip_id,
                        mention.kind,
                        mention.name,
                        confidence=mention.confidence,
                        analyzer=analyzer.name,
                        at_s=mention.at_s,
                    )

            if result.fields:
                ClipRepository.update_fields(
                    session, clip_id, **_clean_fields(result.fields)
                )

            if isinstance(embedding, dict):
                EmbeddingRepository.upsert(
                    session,
                    clip_id,
                    kind=str(embedding["kind"]),
                    model=str(embedding["model"]),
                    dim=int(embedding["dim"]),
                    vector=bytes(embedding["vector"]),
                    norm=float(embedding["norm"]),
                )

        # Sidecar files live outside the transaction: a failed file write must
        # not roll back analysis results that are already valid.
        if thumbnail:
            self._store_thumbnail(clip_id, Path(str(thumbnail)))
        transcript = result.texts.get("transcript")
        if transcript and transcript[0]:
            self._store_transcript(clip_id, transcript[0])

    def _store_thumbnail(self, clip_id: str, source_path: Path) -> None:
        if not source_path.exists():
            return
        target = self.settings.paths.require("thumbnails") / f"{clip_id}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source_path, target)
        except OSError as exc:  # pragma: no cover - disk issue
            log.warning("could not store thumbnail for %s: %s", clip_id[:8], exc)
            return
        with self.db.write_session() as session:
            ClipRepository.update_fields(session, clip_id, thumbnail_path=str(target))

    def _store_transcript(self, clip_id: str, text: str) -> None:
        target = self.settings.paths.require("transcripts") / f"{clip_id}.txt"
        try:
            with atomic_write(target) as handle:
                handle.write(text)
        except OSError as exc:  # pragma: no cover - disk issue
            log.warning("could not store transcript for %s: %s", clip_id[:8], exc)
            return
        with self.db.write_session() as session:
            ClipRepository.update_fields(session, clip_id, transcript_path=str(target))


#: Columns analyzers are permitted to write directly. Anything else in
#: ``fields`` is dropped with a warning rather than blindly applied -- a typo
#: in an analyzer must not silently corrupt the clips table.
WRITABLE_CLIP_FIELDS = frozenset(
    {
        "width", "height", "aspect_ratio", "fps", "video_codec", "audio_codec",
        "video_bitrate", "audio_bitrate", "sample_rate", "channels", "interlaced",
        "quality_score", "quality_notes", "size_bytes", "language", "network",
        "network_confidence", "air_year", "air_year_confidence", "genre", "series",
        "episode_title", "summary", "title", "chromaprint", "has_credits", "has_intro",
        "thumbnail_path", "transcript_path",
    }
)


def _clean_fields(fields: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in fields.items():
        if key in WRITABLE_CLIP_FIELDS:
            cleaned[key] = value
        else:
            log.warning("analyzer tried to write unknown clip field %r; ignored", key)
    return cleaned


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def collect_votes(outcome: AnalysisOutcome) -> list[tuple[Category, float, str]]:
    """Flatten an outcome's evidence into the categorizer's input shape."""
    return [(e.category, e.weight, f"{e.source}: {e.reason}") for e in outcome.evidence]
