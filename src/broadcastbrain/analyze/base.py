"""The analyzer contract.

Every piece of AI in BroadcastBrain is an :class:`Analyzer`: a small object
with a name, a version, a list of backends it needs, and a ``run`` method that
returns data. Nothing else. That uniformity buys three things that matter for
a system meant to last:

* **Selective re-work.** Results are stored per ``(clip, analyzer, version)``.
  Bump one analyzer's version and only that analyzer re-runs, on every clip,
  automatically -- no full reprocessing, ever.
* **Graceful degradation.** A missing backend is a ``skipped`` result, not a
  failure. The clip still finishes; the analyzer runs later when the backend
  appears.
* **Shared expensive inputs.** Frames and analysis audio are extracted once per
  clip by the runner and handed to every analyzer, so adding a tenth vision
  analyzer costs inference time and no extra decoding.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from ..config import Settings
from ..media.ffmpeg import FFmpeg
from ..media.probe import MediaInfo
from ..types import Category, EntityKind

log = logging.getLogger(__name__)


@dataclass
class ClipView:
    """The immutable facts about the clip being analyzed.

    A plain value object rather than the ORM row, so analyzers cannot
    accidentally write to the database and cannot hold a session open across a
    long model inference.
    """

    id: str
    source_id: str
    path: Path
    start_s: float
    end_s: float
    duration_s: float
    seq: int = 0
    media: Optional[MediaInfo] = None
    source_filename: str = ""
    #: Original path of the recording this clip came from. Operators file
    #: material in meaningful folders ("Commercials/2004/"), and that is
    #: evidence no analyzer can recover from the pixels.
    source_path: str = ""
    source_kind: str = "unknown"

    @property
    def midpoint_s(self) -> float:
        return self.duration_s / 2.0


@dataclass
class AnalysisContext:
    """Everything an analyzer is given.

    ``previous`` holds the payloads of analyzers that have already run for this
    clip in this pass *and* those loaded from the database. That is how the
    inference analyzer can reason over OCR text and transcripts without
    re-deriving them, and why ordering (``Analyzer.order``) is part of the
    contract.
    """

    clip: ClipView
    settings: Settings
    ffmpeg: FFmpeg
    workdir: Path
    frames: list[Path] = field(default_factory=list)
    audio_path: Optional[Path] = None
    thumbnail_path: Optional[Path] = None
    previous: dict[str, dict[str, Any]] = field(default_factory=dict)

    def payload_of(self, analyzer: str) -> dict[str, Any]:
        """Payload of an earlier analyzer, or an empty dict."""
        value = self.previous.get(analyzer)
        return value if isinstance(value, dict) else {}

    def text_of(self, analyzer: str, key: str = "text") -> str:
        value = self.payload_of(analyzer).get(key)
        return value if isinstance(value, str) else ""

    def all_text(self) -> str:
        """Transcript + OCR combined -- the evidence pool for text reasoning."""
        parts = [self.text_of("speech", "text"), self.text_of("ocr", "text")]
        return "\n".join(p for p in parts if p)


@dataclass
class EntityMention:
    kind: EntityKind
    name: str
    confidence: float
    at_s: Optional[float] = None


@dataclass
class AnalyzerResult:
    """What an analyzer returns.

    Separating ``payload`` (raw evidence, kept forever) from ``fields``
    (columns to write on the clip) keeps analyzers from fighting over the same
    column: only the inference and categorization stages set the authoritative
    fields, and they do it from the accumulated evidence.
    """

    payload: dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    backend: Optional[str] = None
    entities: list[EntityMention] = field(default_factory=list)
    tags: list[tuple[str, float]] = field(default_factory=list)
    #: kind -> (content, detail) for the ``clip_texts`` table.
    texts: dict[str, tuple[str, Optional[dict[str, Any]]]] = field(default_factory=dict)
    #: Direct column updates on the clip row. Use sparingly.
    fields: dict[str, Any] = field(default_factory=dict)
    #: Classification votes contributed to categorization.
    category_votes: list[tuple[Category, float, str]] = field(default_factory=list)

    def add_entity(
        self, kind: EntityKind, name: str, confidence: float, at_s: Optional[float] = None
    ) -> None:
        name = name.strip()
        if name:
            self.entities.append(EntityMention(kind, name, confidence, at_s))

    def add_tag(self, tag: str, confidence: float = 1.0) -> None:
        tag = tag.strip().lower()
        if tag:
            self.tags.append((tag, confidence))

    def vote(self, category: Category, weight: float, reason: str) -> None:
        self.category_votes.append((category, weight, reason))


class Analyzer(abc.ABC):
    """Base class for every analyzer.

    Subclasses set the class attributes and implement :meth:`run`.
    """

    #: Stable identifier. Used as the database key -- never rename casually.
    name: str = "analyzer"
    #: Bump when the analyzer's output changes meaningfully. Every clip whose
    #: stored version differs is automatically re-analyzed.
    version: str = "1"
    #: Capability names from :mod:`broadcastbrain.capabilities`.
    requires: tuple[str, ...] = ()
    #: Whether this analyzer should hold the GPU slot while it runs.
    needs_gpu: bool = False
    #: Lower runs first. Analyzers that consume others' output run later.
    order: int = 100
    #: Human description, shown by ``bb analyzers``.
    description: str = ""
    #: Declared input needs. The runner extracts frames and analysis audio
    #: once per clip, and only when some analyzer that is actually going to
    #: run has asked for them -- a cached clip costs no decoding at all.
    needs_frames: bool = False
    needs_audio: bool = False

    def applicable(self, ctx: AnalysisContext) -> bool:
        """Whether this analyzer has anything to do for this clip.

        Default: yes. Override to skip cheaply -- for example, speech
        transcription on a 3-second station ID.
        """
        return True

    @abc.abstractmethod
    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        """Analyze the clip.

        Raises:
            BackendUnavailableError: recorded as ``skipped`` and retried later.
            AnalyzerError: recorded as ``failed`` for this analyzer only.
        """

    # -- helpers for subclasses --------------------------------------------

    def check_backends(self) -> None:
        """Raise :class:`BackendUnavailableError` for the first missing backend."""
        from .. import capabilities

        for requirement in self.requires:
            capabilities.require(requirement, self.name)

    def available(self) -> bool:
        from .. import capabilities

        return all(capabilities.has(requirement) for requirement in self.requires)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "requires": list(self.requires),
            "needs_gpu": self.needs_gpu,
            "order": self.order,
            "available": self.available(),
            "description": self.description,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Analyzer {self.name} v{self.version}>"


def sample_frames(frames: Sequence[Path], limit: int) -> list[Path]:
    """Evenly sample up to ``limit`` frames from an extracted frame list.

    Analyzers that are expensive per frame (face recognition) use fewer frames
    than cheap ones (OCR), without anyone re-extracting anything.
    """
    if limit <= 0 or not frames:
        return []
    if len(frames) <= limit:
        return list(frames)
    step = len(frames) / limit
    return [frames[min(len(frames) - 1, int(i * step))] for i in range(limit)]
