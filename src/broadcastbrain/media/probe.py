"""Container inspection via ffprobe.

Two jobs: give the rest of the pipeline a typed, boring view of a file's
technical properties, and decide early whether a file is worth processing at
all. Rejecting a DRM-encrypted VOB here costs a second; discovering it during
analysis costs an hour.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

from ..constants import FFPROBE_TIMEOUT_S
from ..errors import ProbeError, UnsupportedMediaError
from ..types import SourceKind
from ..util.proc import run

log = logging.getLogger(__name__)

#: Common broadcast aspect ratios, used to label clips and to notice the
#: 4:3 -> 16:9 switch that often marks a commercial break in this era.
_STANDARD_ASPECTS: tuple[tuple[float, str], ...] = (
    (4 / 3, "4:3"),
    (16 / 9, "16:9"),
    (16 / 10, "16:10"),
    (2.35, "2.35:1"),
    (1.85, "1.85:1"),
    (1.0, "1:1"),
)


@dataclass
class StreamInfo:
    """One decoded stream's properties."""

    index: int
    codec_type: str
    codec_name: str = ""
    codec_long_name: str = ""
    bit_rate: Optional[int] = None
    language: Optional[str] = None
    # video
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    pix_fmt: Optional[str] = None
    field_order: Optional[str] = None
    display_aspect_ratio: Optional[str] = None
    nb_frames: Optional[int] = None
    # audio
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    channel_layout: Optional[str] = None

    @property
    def is_video(self) -> bool:
        return self.codec_type == "video"

    @property
    def is_audio(self) -> bool:
        return self.codec_type == "audio"

    @property
    def interlaced(self) -> bool:
        return bool(self.field_order and self.field_order not in ("progressive", "unknown"))


@dataclass
class MediaInfo:
    """Everything the pipeline needs to know about a media file."""

    path: str
    container: str = ""
    duration_s: float = 0.0
    size_bytes: int = 0
    bit_rate: Optional[int] = None
    streams: list[StreamInfo] = field(default_factory=list)
    chapters: list[dict[str, Any]] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    raw: Optional[dict[str, Any]] = None

    # -- convenience accessors ---------------------------------------------

    @property
    def video(self) -> Optional[StreamInfo]:
        """The primary video stream (largest by pixel count)."""
        videos = [s for s in self.streams if s.is_video and (s.width or 0) > 0]
        if not videos:
            return None
        return max(videos, key=lambda s: (s.width or 0) * (s.height or 0))

    @property
    def audio(self) -> Optional[StreamInfo]:
        audios = [s for s in self.streams if s.is_audio]
        if not audios:
            return None
        return max(audios, key=lambda s: (s.channels or 0, s.bit_rate or 0))

    @property
    def has_video(self) -> bool:
        return self.video is not None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    @property
    def width(self) -> Optional[int]:
        return self.video.width if self.video else None

    @property
    def height(self) -> Optional[int]:
        return self.video.height if self.video else None

    @property
    def fps(self) -> Optional[float]:
        return self.video.fps if self.video else None

    @property
    def video_codec(self) -> Optional[str]:
        return self.video.codec_name if self.video else None

    @property
    def audio_codec(self) -> Optional[str]:
        return self.audio.codec_name if self.audio else None

    @property
    def interlaced(self) -> bool:
        return bool(self.video and self.video.interlaced)

    @property
    def aspect_ratio(self) -> Optional[str]:
        """Labelled aspect ratio, preferring the container's DAR."""
        video = self.video
        if video is None:
            return None
        if video.display_aspect_ratio and ":" in video.display_aspect_ratio:
            return video.display_aspect_ratio
        if not video.width or not video.height:
            return None
        return classify_aspect(video.width / video.height)

    @property
    def pixel_count(self) -> int:
        video = self.video
        return (video.width or 0) * (video.height or 0) if video else 0

    def guess_source_kind(self) -> SourceKind:
        """Infer provenance from technical fingerprints.

        These heuristics are coarse by design -- they feed duplicate ranking
        and year estimation as *evidence*, never as a verdict.
        """
        video = self.video
        if video is None:
            return SourceKind.UNKNOWN
        container = self.container.lower()
        codec = (video.codec_name or "").lower()
        height = video.height or 0

        if codec in {"mpeg2video"} and container in {"mpeg", "vob", "dvd", "mpegts", "mpeg2video"}:
            # 480i MPEG-2 in a VOB/TS is either a DVD or a DVR capture.
            return SourceKind.DVD if "vob" in container or "dvd" in container else SourceKind.DVR
        if video.interlaced and height in (480, 486, 576) and codec in {"dvvideo", "mpeg2video"}:
            return SourceKind.VHS
        if codec in {"h264", "hevc", "vp9", "av1"} and height >= 480:
            return SourceKind.DIGITAL_CAPTURE
        return SourceKind.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        data["aspect_ratio"] = self.aspect_ratio
        data["source_kind_guess"] = self.guess_source_kind().value
        return data


def classify_aspect(ratio: float, *, tolerance: float = 0.05) -> str:
    """Snap a numeric aspect ratio to a standard label, or format it."""
    for value, label in _STANDARD_ASPECTS:
        if abs(ratio - value) <= tolerance:
            return label
    return f"{ratio:.2f}:1"


def _parse_fraction(value: Optional[str]) -> Optional[float]:
    """Parse ffprobe's ``30000/1001`` style rates safely."""
    if not value or value in ("0/0", "N/A"):
        return None
    try:
        if "/" in value:
            fraction = Fraction(value)
            return float(fraction) if fraction.denominator else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def probe(
    path: Path,
    *,
    ffprobe_binary: str = "ffprobe",
    timeout_s: int = FFPROBE_TIMEOUT_S,
    count_frames: bool = False,
) -> MediaInfo:
    """Run ffprobe and return typed media information.

    Args:
        count_frames: Decode the whole file to get an exact frame count.
            Off by default -- it is minutes of work on a multi-hour capture and
            the pipeline never needs exact counts.

    Raises:
        ProbeError: the file could not be parsed.
        UnsupportedMediaError: parsed fine, but has nothing we can use.
    """
    if not path.exists():
        raise ProbeError(f"file does not exist: {path}")

    args: list[str] = [
        ffprobe_binary,
        "-v", "error",
        "-hide_banner",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
    ]
    if count_frames:
        args += ["-count_frames"]
    args += [str(path)]

    result = run(args, timeout_s=timeout_s, check=False, stderr_tail_lines=40)
    if result.returncode != 0 or not result.stdout.strip():
        raise ProbeError(f"ffprobe failed for {path.name}: {result.stderr.strip()[:500]}")

    try:
        payload: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ffprobe emitted invalid JSON for {path.name}: {exc}") from exc

    info = _build_media_info(path, payload)
    if not info.has_video and not info.has_audio:
        raise UnsupportedMediaError(f"{path.name} contains no decodable audio or video")
    return info


def _build_media_info(path: Path, payload: dict[str, Any]) -> MediaInfo:
    fmt = payload.get("format") or {}
    streams: list[StreamInfo] = []

    for raw in payload.get("streams") or []:
        codec_type = str(raw.get("codec_type") or "")
        stream = StreamInfo(
            index=_as_int(raw.get("index")) or 0,
            codec_type=codec_type,
            codec_name=str(raw.get("codec_name") or ""),
            codec_long_name=str(raw.get("codec_long_name") or ""),
            bit_rate=_as_int(raw.get("bit_rate")),
            language=(raw.get("tags") or {}).get("language"),
        )
        if codec_type == "video":
            stream.width = _as_int(raw.get("width"))
            stream.height = _as_int(raw.get("height"))
            stream.pix_fmt = raw.get("pix_fmt")
            stream.field_order = raw.get("field_order")
            stream.display_aspect_ratio = raw.get("display_aspect_ratio")
            stream.nb_frames = _as_int(raw.get("nb_frames") or raw.get("nb_read_frames"))
            # avg_frame_rate is 0/0 for VFR; r_frame_rate is the safer fallback.
            stream.fps = _parse_fraction(raw.get("avg_frame_rate")) or _parse_fraction(
                raw.get("r_frame_rate")
            )
        elif codec_type == "audio":
            stream.sample_rate = _as_int(raw.get("sample_rate"))
            stream.channels = _as_int(raw.get("channels"))
            stream.channel_layout = raw.get("channel_layout")
        streams.append(stream)

    duration = _parse_fraction(fmt.get("duration")) or 0.0
    if duration <= 0:
        # Some TS/VOB captures have no container duration; fall back to the
        # longest stream's own duration tag.
        for raw in payload.get("streams") or []:
            candidate = _parse_fraction(raw.get("duration")) or 0.0
            duration = max(duration, candidate)

    return MediaInfo(
        path=str(path),
        container=str(fmt.get("format_name") or path.suffix.lstrip(".")),
        duration_s=duration,
        size_bytes=_as_int(fmt.get("size")) or (path.stat().st_size if path.exists() else 0),
        bit_rate=_as_int(fmt.get("bit_rate")),
        streams=streams,
        chapters=list(payload.get("chapters") or []),
        tags={str(k): str(v) for k, v in (fmt.get("tags") or {}).items()},
        raw=payload,
    )


def quality_score(info: MediaInfo, *, source_trust: float = 0.5) -> float:
    """A 0-1 technical quality score used to pick which duplicate to keep.

    Deliberately simple and explainable: resolution and bitrate dominate,
    frame rate and audio contribute, and provenance breaks ties. An operator
    can read the number and know why one copy won.
    """
    from ..constants import QUALITY_WEIGHTS

    video = info.video
    if video is None:
        return 0.0

    pixels = (video.width or 0) * (video.height or 0)
    # 1920x1080 is the practical ceiling for this era's material.
    pixel_score = min(1.0, pixels / (1920 * 1080))

    bitrate = video.bit_rate or info.bit_rate or 0
    bitrate_score = min(1.0, bitrate / 8_000_000) if bitrate else 0.3

    fps = video.fps or 0
    fps_score = min(1.0, fps / 30.0) if fps else 0.5

    audio = info.audio
    audio_score = min(1.0, (audio.bit_rate or 128_000) / 256_000) if audio else 0.0

    weights = QUALITY_WEIGHTS
    score = (
        pixel_score * weights["pixels"]
        + bitrate_score * weights["bitrate"]
        + fps_score * weights["fps"]
        + audio_score * weights["audio_bitrate"]
        + 1.0 * weights["duration_completeness"]
        + source_trust * weights["source_trust"]
    )
    return round(min(1.0, max(0.0, score)), 4)
