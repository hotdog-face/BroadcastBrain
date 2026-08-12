"""Shared enums and value objects.

These are the vocabulary of the pipeline. They are deliberately free of any
database, ffmpeg or model imports so every layer can depend on them without
creating a cycle.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Optional


class _StrEnum(str, Enum):
    """`str`-backed enum that serializes to its value (py3.9 compatible)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class Stage(_StrEnum):
    """Pipeline stages, in execution order.

    A source or clip records the last stage it *completed*. Resuming means
    re-enqueuing work for whatever comes next -- never redoing what is done.
    """

    DISCOVERED = "discovered"
    PROBED = "probed"
    SEGMENTED = "segmented"
    EXTRACTED = "extracted"
    ANALYZED = "analyzed"
    FINGERPRINTED = "fingerprinted"
    DEDUPED = "deduped"
    CATEGORIZED = "categorized"
    MATERIALIZED = "materialized"
    EXPORTED = "exported"

    @property
    def order(self) -> int:
        return _STAGE_ORDER[self]


_STAGE_ORDER: dict[Stage, int] = {s: i for i, s in enumerate(Stage)}


class JobKind(_StrEnum):
    """Unit of queued work. One kind maps to exactly one handler."""

    INGEST_SOURCE = "ingest_source"
    SEGMENT_SOURCE = "segment_source"
    ANALYZE_CLIP = "analyze_clip"
    FINGERPRINT_CLIP = "fingerprint_clip"
    DEDUPE_CLIP = "dedupe_clip"
    CATEGORIZE_CLIP = "categorize_clip"
    MATERIALIZE_CLIP = "materialize_clip"
    EXPORT_LIBRARY = "export_library"
    DOWNLOAD_ITEM = "download_item"


class JobState(_StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"  # will be retried
    DEAD = "dead"  # exhausted retries; needs operator attention
    CANCELLED = "cancelled"


class SourceStatus(_StrEnum):
    DISCOVERED = "discovered"
    PROCESSING = "processing"
    COMPLETE = "complete"
    REJECTED = "rejected"
    FAILED = "failed"


class ClipStatus(_StrEnum):
    NEW = "new"
    PROCESSING = "processing"
    READY = "ready"
    DUPLICATE = "duplicate"
    FAILED = "failed"


class Category(_StrEnum):
    """What a clip *is*. `UNKNOWN` is a first-class outcome, never a discard."""

    COMMERCIAL = "commercial"
    TV_EPISODE = "tv_episode"
    MOVIE = "movie"
    PROMO = "promo"
    NETWORK_BUMPER = "network_bumper"
    STATION_ID = "station_id"
    MUSIC_VIDEO = "music_video"
    INFOMERCIAL = "infomercial"
    PPV_PREVIEW = "ppv_preview"
    WEATHER = "weather"
    NEWS_BREAK = "news_break"
    TV_GUIDE = "tv_guide"
    INTRO = "intro"
    CREDITS = "credits"
    UNKNOWN = "unknown"


#: Categories Fieldstation can use as interstitial filler between programs.
INTERSTITIAL_CATEGORIES: frozenset[Category] = frozenset(
    {
        Category.COMMERCIAL,
        Category.PROMO,
        Category.NETWORK_BUMPER,
        Category.STATION_ID,
        Category.PPV_PREVIEW,
    }
)

#: Categories that are standalone programming (they get scheduled, not filled).
PROGRAM_CATEGORIES: frozenset[Category] = frozenset(
    {
        Category.TV_EPISODE,
        Category.MOVIE,
        Category.MUSIC_VIDEO,
        Category.INFOMERCIAL,
        Category.NEWS_BREAK,
        Category.WEATHER,
    }
)


class EntityKind(_StrEnum):
    PERSON = "person"
    BRAND = "brand"
    COMPANY = "company"
    PRODUCT = "product"
    OBJECT = "object"
    SHOW = "show"
    NETWORK = "network"
    SONG = "song"
    ARTIST = "artist"
    PLACE = "place"


class AnalysisStatus(_StrEnum):
    OK = "ok"
    SKIPPED = "skipped"  # backend unavailable / not applicable -- retryable later
    FAILED = "failed"


class SignalKind(_StrEnum):
    """Segmentation evidence types. Values match `constants.SIGNAL_WEIGHTS`."""

    BLACK = "black"
    SILENCE = "silence"
    SCENE = "scene"
    FREEZE = "freeze"
    AUDIO_CHANGE = "audio_change"
    LOGO_CHANGE = "logo_change"
    ASPECT_CHANGE = "aspect_change"
    LOUDNESS_JUMP = "loudness_jump"
    CREDITS = "credits"
    FADE = "fade"


class DuplicateMethod(_StrEnum):
    EXACT_SHA256 = "sha256"
    PERCEPTUAL = "phash"
    AUDIO_FINGERPRINT = "chromaprint"
    COMPOSITE = "composite"


class SourceKind(_StrEnum):
    """Provenance of a source recording -- feeds quality scoring and priors."""

    UNKNOWN = "unknown"
    VHS = "vhs"
    DVD = "dvd"
    DVR = "dvr"
    DIGITAL_CAPTURE = "digital_capture"
    INTERNET_ARCHIVE = "internet_archive"


#: How much we trust each provenance when ranking duplicates (0-1).
SOURCE_TRUST: dict[SourceKind, float] = {
    SourceKind.DVD: 1.00,
    SourceKind.DIGITAL_CAPTURE: 0.90,
    SourceKind.DVR: 0.80,
    SourceKind.INTERNET_ARCHIVE: 0.65,
    SourceKind.VHS: 0.55,
    SourceKind.UNKNOWN: 0.50,
}


@dataclasses.dataclass(frozen=True)
class Signal:
    """One piece of segmentation evidence at a point in time.

    `strength` is a 0-1 confidence *within* the signal kind; the fuser
    multiplies it by the kind's weight from :mod:`constants`.
    """

    kind: SignalKind
    time_s: float
    strength: float = 1.0
    duration_s: float = 0.0
    detail: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "time_s": round(self.time_s, 3),
            "strength": round(self.strength, 4),
            "duration_s": round(self.duration_s, 3),
            "detail": self.detail,
        }


@dataclasses.dataclass(frozen=True)
class Boundary:
    """A fused cut point: several signals agreeing at roughly one timestamp."""

    time_s: float
    score: float
    kinds: tuple[SignalKind, ...]

    @property
    def is_strong(self) -> bool:
        from .constants import BOUNDARY_STRONG_SCORE

        return self.score >= BOUNDARY_STRONG_SCORE

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_s": round(self.time_s, 3),
            "score": round(self.score, 4),
            "kinds": [k.value for k in self.kinds],
        }


@dataclasses.dataclass(frozen=True)
class Segment:
    """A proposed clip: a span of a source plus the evidence that produced it."""

    index: int
    start_s: float
    end_s: float
    start_score: float = 0.0
    end_score: float = 0.0
    signal_kinds: tuple[SignalKind, ...] = ()

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
            "start_score": round(self.start_score, 4),
            "end_score": round(self.end_score, 4),
            "signal_kinds": [k.value for k in self.signal_kinds],
        }


@dataclasses.dataclass(frozen=True)
class Evidence:
    """A single classification vote produced by a rule or analyzer.

    Keeping votes as data (rather than mutating a label in place) is what makes
    categorization explainable: every clip stores why it was labelled.
    """

    category: Category
    weight: float
    reason: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "weight": round(self.weight, 4),
            "reason": self.reason,
            "source": self.source,
        }
