"""ffmpeg operations: cutting clips, thumbnails, frames and analysis audio.

Everything that touches ffmpeg lives behind :class:`FFmpeg` so the pipeline
never assembles a command line itself. That gives one place to add hardware
acceleration, one place to get seeking right, and one seam for tests to stub.

The seeking detail matters more than it looks: ``-ss`` *before* ``-i`` seeks by
keyframe (instant, but can land up to a GOP early), while ``-ss`` *after*
``-i`` decodes from the start (exact, but O(position)). Cutting a clip an hour
into a six-hour capture the slow way would take minutes. :meth:`cut` uses the
fast form, optionally refined by an exact pass, which is why segmentation of a
multi-hour recording finishes in reasonable time.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from .. import capabilities
from ..config import Settings
from ..constants import (
    ANALYSIS_AUDIO_CHANNELS,
    ANALYSIS_AUDIO_SAMPLE_RATE,
    FFMPEG_SEGMENT_TIMEOUT_S,
    THUMBNAIL_QUALITY,
    THUMBNAIL_WIDTH,
    VISION_MIN_FRAME_SPACING_S,
)
from ..errors import MediaError, SubprocessError
from ..util.atomic import atomic_move
from ..util.proc import CompletedCommand, run

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CutResult:
    path: Path
    start_s: float
    end_s: float
    stream_copied: bool
    duration_s: float


class FFmpeg:
    """Thin, typed wrapper around the ffmpeg binary."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.binary = settings.ffmpeg_binary
        self.nice = settings.workers.subprocess_nice
        self._hwaccel: Optional[str] = (
            capabilities.preferred_hwaccel(settings.workers.hwaccel)
            if settings.workers.use_gpu
            else None
        )

    # -- helpers ------------------------------------------------------------

    @property
    def hwaccel(self) -> Optional[str]:
        return self._hwaccel

    def _base(self, *, hwaccel: bool = False) -> list[str]:
        args = [self.binary, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
        if hwaccel and self._hwaccel:
            args += ["-hwaccel", self._hwaccel]
        return args

    def _run(
        self, args: Sequence[str], *, timeout_s: float, on_line: Optional[object] = None
    ) -> CompletedCommand:
        return run(
            args,
            timeout_s=timeout_s,
            nice=self.nice,
            stderr_tail_lines=60,
            log_command=self.settings.logging.log_subprocess_commands,
            on_line=on_line,  # type: ignore[arg-type]
        )

    # -- cutting ------------------------------------------------------------

    def cut(
        self,
        source: Path,
        target: Path,
        *,
        start_s: float,
        end_s: float,
        reencode: Optional[bool] = None,
        timeout_s: float = FFMPEG_SEGMENT_TIMEOUT_S,
    ) -> CutResult:
        """Extract ``[start_s, end_s)`` from ``source`` into ``target``.

        Tries a stream copy first: it is roughly a hundred times faster than
        re-encoding and is lossless. Stream copies land on keyframe boundaries,
        so when frame accuracy is required (``segmentation.reencode``) or the
        copy produces an unusable file, it falls back to a re-encode.

        The output is written to a temporary name and moved into place, so a
        killed process never leaves a plausible-looking truncated clip.
        """
        duration = max(0.0, end_s - start_s)
        if duration <= 0:
            raise MediaError(f"refusing to cut a non-positive span: {start_s} -> {end_s}")

        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.parent / f".{target.name}.partial{target.suffix or '.mkv'}"
        want_reencode = self.settings.segmentation.reencode if reencode is None else reencode

        if not want_reencode:
            try:
                self._run(self._copy_cut_args(source, temp, start_s, duration), timeout_s=timeout_s)
                if _looks_usable(temp):
                    return CutResult(
                        path=atomic_move(temp, target, overwrite=True),
                        start_s=start_s,
                        end_s=end_s,
                        stream_copied=True,
                        duration_s=duration,
                    )
                log.debug("stream copy produced an unusable clip; re-encoding %s", target.name)
            except SubprocessError as exc:
                log.debug("stream copy failed (%s); re-encoding %s", exc.returncode, target.name)
            _unlink(temp)

        self._run(self._reencode_cut_args(source, temp, start_s, duration), timeout_s=timeout_s)
        if not _looks_usable(temp):
            _unlink(temp)
            raise MediaError(f"ffmpeg produced no usable output for {target.name}")
        return CutResult(
            path=atomic_move(temp, target, overwrite=True),
            start_s=start_s,
            end_s=end_s,
            stream_copied=False,
            duration_s=duration,
        )

    def _copy_cut_args(self, source: Path, target: Path, start_s: float, duration: float) -> list[str]:
        return [
            *self._base(),
            "-ss", f"{start_s:.3f}",          # fast keyframe seek, before -i
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-map", "0",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            "-max_muxing_queue_size", "2048",
            str(target),
        ]

    def _reencode_cut_args(
        self, source: Path, target: Path, start_s: float, duration: float
    ) -> list[str]:
        seg = self.settings.segmentation
        return [
            *self._base(hwaccel=True),
            "-ss", f"{max(0.0, start_s - 5.0):.3f}",   # coarse seek...
            "-i", str(source),
            "-ss", f"{min(5.0, start_s):.3f}",         # ...then exact trim
            "-t", f"{duration:.3f}",
            "-c:v", seg.video_encoder,
            "-crf", str(seg.encoder_crf),
            "-preset", seg.encoder_preset,
            "-c:a", seg.audio_encoder,
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-max_muxing_queue_size", "2048",
            str(target),
        ]

    # -- derived assets -----------------------------------------------------

    def thumbnail(
        self,
        source: Path,
        target: Path,
        *,
        at_s: float,
        width: int = THUMBNAIL_WIDTH,
        timeout_s: float = 120,
    ) -> Optional[Path]:
        """Grab a single representative frame as a JPEG.

        Returns ``None`` rather than raising: a missing thumbnail must never
        fail a clip.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.parent / f".{target.name}.partial.jpg"
        args = [
            *self._base(),
            "-ss", f"{max(0.0, at_s):.3f}",
            "-i", str(source),
            "-frames:v", "1",
            "-vf", f"scale={width}:-2:flags=lanczos",
            "-q:v", str(THUMBNAIL_QUALITY),
            str(temp),
        ]
        try:
            self._run(args, timeout_s=timeout_s)
        except (SubprocessError, MediaError) as exc:
            log.debug("thumbnail failed for %s: %s", source.name, exc)
            _unlink(temp)
            return None
        if not _looks_usable(temp, min_bytes=512):
            _unlink(temp)
            return None
        return atomic_move(temp, target, overwrite=True)

    def extract_audio(
        self,
        source: Path,
        target: Path,
        *,
        sample_rate: int = ANALYSIS_AUDIO_SAMPLE_RATE,
        channels: int = ANALYSIS_AUDIO_CHANNELS,
        timeout_s: float = 1800,
    ) -> Optional[Path]:
        """Mono 16 kHz WAV -- what Whisper and Chromaprint both want."""
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.parent / f".{target.name}.partial.wav"
        args = [
            *self._base(),
            "-i", str(source),
            "-vn",
            "-ac", str(channels),
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(temp),
        ]
        try:
            self._run(args, timeout_s=timeout_s)
        except SubprocessError as exc:
            log.debug("audio extraction failed for %s: %s", source.name, exc)
            _unlink(temp)
            return None
        if not _looks_usable(temp, min_bytes=1024):
            _unlink(temp)
            return None
        return atomic_move(temp, target, overwrite=True)

    def extract_frames(
        self,
        source: Path,
        directory: Path,
        *,
        count: int,
        duration_s: float,
        width: int = 640,
        timeout_s: float = 900,
    ) -> list[Path]:
        """Sample ``count`` evenly spaced frames into ``directory``.

        Frames are the shared input for OCR, logo, face and object analysis, so
        they are extracted once per clip and reused by every vision analyzer
        rather than each one decoding the clip again.
        """
        if count <= 0 or duration_s <= 0:
            return []
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("frame_*.jpg"):
            _unlink(stale)

        spacing = max(VISION_MIN_FRAME_SPACING_S, duration_s / max(1, count))
        fps_expr = 1.0 / spacing
        args = [
            *self._base(hwaccel=True),
            "-i", str(source),
            "-vf", f"fps={fps_expr:.6f},scale={width}:-2:flags=bilinear",
            "-frames:v", str(count),
            "-q:v", "4",
            str(directory / "frame_%04d.jpg"),
        ]
        try:
            self._run(args, timeout_s=timeout_s)
        except SubprocessError as exc:
            log.debug("frame extraction failed for %s: %s", source.name, exc)
            return []
        return sorted(directory.glob("frame_*.jpg"))

    def frame_at(
        self, source: Path, target: Path, *, at_s: float, width: int = 640, timeout_s: float = 120
    ) -> Optional[Path]:
        """Single frame at a precise time (used for logo-region sampling)."""
        return self.thumbnail(source, target, at_s=at_s, width=width, timeout_s=timeout_s)

    # -- diagnostics --------------------------------------------------------

    def version(self) -> str:
        try:
            result = run([self.binary, "-version"], timeout_s=30, check=False)
        except Exception as exc:
            return f"unavailable: {exc}"
        first = result.stdout.splitlines()[0] if result.stdout else ""
        return first or "unknown"

    def available(self) -> bool:
        return shutil.which(self.binary) is not None


def _looks_usable(path: Path, *, min_bytes: int = 4096) -> bool:
    """A file exists and is big enough to plausibly contain media."""
    try:
        return path.exists() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()
