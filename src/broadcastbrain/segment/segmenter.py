"""Segmentation orchestration: source in, segments out.

This is the thin layer that wires the collector to the fuser and applies the
policies that need to know about the *kind* of recording being processed. A
DVD title of a single movie should not be shredded into scene-length clips; a
six-hour cable capture should be. Those decisions live here rather than in the
fusion maths, which stays pure and testable.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from ..config import Settings
from ..errors import SegmentationError
from ..media.probe import MediaInfo
from ..types import Boundary, Segment, SourceKind
from .fusion import FusionConfig, adapt_weights, describe, fuse
from .signals import SignalCollector, SignalReport

log = logging.getLogger(__name__)

#: Sources shorter than this are almost always a single item already (a
#: downloaded commercial reel entry, one music video). Segmenting them adds
#: risk without adding value.
SINGLE_CLIP_MAX_SECONDS = 90.0

#: When a source yields fewer than this many segments per hour, segmentation
#: probably failed to find real boundaries. The result is still kept -- it is
#: recorded as low confidence so ``bb resegment`` can revisit it.
MIN_SEGMENTS_PER_HOUR = 2.0


@dataclass
class SegmentationResult:
    """Segments plus the full audit trail of how they were derived."""

    segments: list[Segment]
    boundaries: list[Boundary]
    report: SignalReport
    confidence: float
    strategy: str
    #: Signals whose weight was reduced because they fired at a noise rate.
    adaptations: list[str] = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        data = describe(self.segments, self.boundaries)
        data.update(
            {
                "signals": self.report.summary(),
                "confidence": round(self.confidence, 3),
                "strategy": self.strategy,
                "adaptations": self.adaptations,
            }
        )
        return data


class Segmenter:
    """Splits a recording into logical clips using every available signal."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collector = SignalCollector(settings)
        self.config = FusionConfig.from_settings(settings)

    def segment(
        self,
        source: Path,
        info: MediaInfo,
        *,
        source_kind: SourceKind = SourceKind.UNKNOWN,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> SegmentationResult:
        """Produce segments for one source file.

        Raises:
            SegmentationError: the source has no usable duration.
        """
        duration = info.duration_s
        if duration <= 0:
            raise SegmentationError(f"{source.name} reports no duration; cannot segment")

        if not self.settings.segmentation.enabled:
            return self._whole_file(info, "segmentation disabled")

        if duration <= SINGLE_CLIP_MAX_SECONDS:
            return self._whole_file(info, "short source kept whole")

        report = self.collector.collect(source, info, progress=progress)
        config = self._config_for(source_kind, duration)

        # Detectors that reacted to the *content* rather than to programme
        # structure get their weight cut before fusion. On live-action
        # broadcast this changes nothing; on animation it is the difference
        # between shot-length fragments and real clips.
        adjusted, adaptations = adapt_weights(config.weights, report.counts, duration)
        if adaptations:
            config = replace(config, weights=adjusted)
            log.info("%s: adapted noisy signals -- %s", source.name, "; ".join(adaptations))

        segments, boundaries = fuse(report.signals, duration, config)

        if not segments:
            log.warning("%s: fusion produced no segments; keeping the file whole", source.name)
            return self._whole_file(info, "no boundaries found", report=report)

        confidence = self._confidence(segments, boundaries, duration, report)
        strategy = f"multi-signal ({'degraded' if report.degraded else 'full'})"
        if adaptations:
            strategy += " + adaptive weighting"
        log.info(
            "%s: %d segment(s), confidence %.2f, %s",
            source.name, len(segments), confidence, strategy,
        )
        return SegmentationResult(
            segments=segments,
            boundaries=boundaries,
            report=report,
            confidence=confidence,
            strategy=strategy,
            adaptations=adaptations,
        )

    # -- policy -------------------------------------------------------------

    def _config_for(self, source_kind: SourceKind, duration_s: float) -> FusionConfig:
        """Adapt fusion thresholds to the material.

        VHS captures fade through grey rather than hitting true black and are
        noisy enough to trip scene detection constantly, so their boundary bar
        is lowered slightly and their minimum segment length raised. DVD titles
        are usually one program, so the maximum segment length is relaxed to
        avoid chopping a film at its quiet moments.
        """
        config = self.config
        if source_kind is SourceKind.VHS:
            return FusionConfig(
                cluster_window_s=config.cluster_window_s * 1.4,
                min_boundary_score=config.min_boundary_score * 0.85,
                min_segment_s=max(config.min_segment_s, 5.0),
                max_segment_s=config.max_segment_s,
                weights=config.weights,
                grid_bonus=config.grid_bonus,
            )
        if source_kind is SourceKind.DVD and duration_s > 3600:
            return FusionConfig(
                cluster_window_s=config.cluster_window_s,
                min_boundary_score=config.min_boundary_score * 1.25,
                min_segment_s=max(config.min_segment_s, 30.0),
                max_segment_s=config.max_segment_s * 2,
                weights=config.weights,
                grid_bonus=0.0,  # a DVD has no broadcast ad grid
            )
        return config

    def _confidence(
        self,
        segments: list[Segment],
        boundaries: list[Boundary],
        duration_s: float,
        report: SignalReport,
    ) -> float:
        """How much to trust this segmentation, as a 0-1 score.

        Used to decide whether a source needs a human eye, and surfaced in the
        nightly report. It rewards strong, diverse evidence and penalizes
        implausibly sparse results.
        """
        if not boundaries:
            return 0.1
        strong = sum(1 for b in boundaries if b.is_strong)
        strong_ratio = strong / len(boundaries)

        hours = max(duration_s / 3600.0, 1e-6)
        density = len(segments) / hours
        density_score = min(1.0, density / MIN_SEGMENTS_PER_HOUR) if density < MIN_SEGMENTS_PER_HOUR else 1.0

        distinct_kinds = len({kind for boundary in boundaries for kind in boundary.kinds})
        diversity = min(1.0, distinct_kinds / 4.0)

        penalty = 0.75 if report.degraded else 1.0
        return round(
            min(1.0, (0.45 * strong_ratio + 0.3 * density_score + 0.25 * diversity) * penalty), 3
        )

    def _whole_file(
        self, info: MediaInfo, reason: str, *, report: Optional[SignalReport] = None
    ) -> SegmentationResult:
        """Treat the whole source as one clip. Never a failure -- just a shape."""
        duration = info.duration_s
        segment = Segment(index=0, start_s=0.0, end_s=duration)
        return SegmentationResult(
            segments=[segment],
            boundaries=[],
            report=report or SignalReport(duration_s=duration, note=reason),
            confidence=0.5,
            strategy=f"whole-file ({reason})",
        )
