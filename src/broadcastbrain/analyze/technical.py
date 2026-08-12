"""Technical inspection and broadcast-quality estimation.

Runs first, needs no AI backend at all, and populates the columns everything
downstream depends on: resolution, codecs, aspect ratio, and a broadcast
quality score. It also produces the thumbnail, because the frame chosen for a
thumbnail is the same frame worth showing in the dashboard.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..constants import THUMBNAIL_POSITION_RATIO
from ..errors import AnalyzerError, MediaError
from ..media.probe import MediaInfo, probe, quality_score
from ..types import SOURCE_TRUST, SourceKind
from .base import AnalysisContext, Analyzer, AnalyzerResult
from .registry import register

log = logging.getLogger(__name__)

#: Height -> label. Broadcast material in this era is 480i/480p almost
#: universally; anything above is an upscale or a later re-encode.
_RESOLUTION_TIERS: tuple[tuple[int, str], ...] = (
    (1080, "hd-1080"),
    (720, "hd-720"),
    (480, "sd-480"),
    (360, "low-360"),
    (0, "very-low"),
)


@register
class TechnicalAnalyzer(Analyzer):
    """Probe the cut clip and score its technical quality."""

    name = "technical"
    version = "1"
    requires = ()
    order = 5
    description = "Container/codec inspection, aspect ratio, quality scoring, thumbnail"

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        try:
            info: MediaInfo = probe(ctx.clip.path, ffprobe_binary=ctx.settings.ffprobe_binary)
        except MediaError as exc:
            raise AnalyzerError(self.name, str(exc)) from exc

        source_kind = _source_kind(ctx.clip.source_kind)
        trust = SOURCE_TRUST.get(source_kind, 0.5)
        score = quality_score(info, source_trust=trust)
        tier = _resolution_tier(info.height or 0)

        result = AnalyzerResult(backend="ffprobe", confidence=1.0)
        result.payload = {
            "container": info.container,
            "duration_s": round(info.duration_s, 3),
            "resolution": f"{info.width or 0}x{info.height or 0}",
            "resolution_tier": tier,
            "aspect_ratio": info.aspect_ratio,
            "fps": round(info.fps, 3) if info.fps else None,
            "video_codec": info.video_codec,
            "audio_codec": info.audio_codec,
            "interlaced": info.interlaced,
            "bit_rate": info.bit_rate,
            "has_audio": info.has_audio,
            "quality_score": score,
            "quality_factors": _quality_factors(info, trust),
            "source_kind": source_kind.value,
        }

        video = info.video
        audio = info.audio
        result.fields = {
            "width": info.width,
            "height": info.height,
            "aspect_ratio": info.aspect_ratio,
            "fps": info.fps,
            "video_codec": info.video_codec,
            "audio_codec": info.audio_codec,
            "video_bitrate": video.bit_rate if video else None,
            "audio_bitrate": audio.bit_rate if audio else None,
            "sample_rate": audio.sample_rate if audio else None,
            "channels": audio.channels if audio else None,
            "interlaced": info.interlaced,
            "quality_score": score,
            "quality_notes": _quality_factors(info, trust),
            "size_bytes": info.size_bytes,
        }

        result.add_tag(tier, 0.9)
        if info.aspect_ratio:
            result.add_tag(f"aspect-{info.aspect_ratio.replace(':', '-')}", 0.8)
        if info.interlaced:
            result.add_tag("interlaced", 0.9)
        if not info.has_audio:
            result.add_tag("silent", 1.0)

        thumbnail = self._make_thumbnail(ctx)
        if thumbnail is not None:
            result.fields["thumbnail_path"] = str(thumbnail)
            result.payload["thumbnail"] = str(thumbnail)
        return result

    def _make_thumbnail(self, ctx: AnalysisContext) -> Optional[Any]:
        """Grab a frame from a third of the way in.

        Not the first frame: clips very often open on black or a fade, and a
        library full of black thumbnails is useless for browsing.
        """
        target = ctx.workdir / "thumbnail.jpg"
        at = max(0.5, ctx.clip.duration_s * THUMBNAIL_POSITION_RATIO)
        return ctx.ffmpeg.thumbnail(
            ctx.clip.path, target, at_s=at, width=ctx.settings.analysis.thumbnail_width
        )


def _resolution_tier(height: int) -> str:
    for minimum, label in _RESOLUTION_TIERS:
        if height >= minimum:
            return label
    return "unknown"


def _source_kind(value: str) -> SourceKind:
    try:
        return SourceKind(value)
    except ValueError:
        return SourceKind.UNKNOWN


def _quality_factors(info: MediaInfo, trust: float) -> dict[str, Any]:
    """The inputs to the quality score, stored so a ranking can be explained."""
    video = info.video
    return {
        "pixels": (video.width or 0) * (video.height or 0) if video else 0,
        "video_bitrate": video.bit_rate if video else None,
        "container_bitrate": info.bit_rate,
        "fps": video.fps if video else None,
        "audio_channels": info.audio.channels if info.audio else 0,
        "interlaced": info.interlaced,
        "source_trust": trust,
    }
