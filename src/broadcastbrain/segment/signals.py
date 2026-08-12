"""Multi-signal evidence collection.

Black-frame detection alone is not enough for real archived television. A VHS
capture fades to grey rather than black; a station that segues straight from a
promo into a show has no black at all; a DVD rip may have had its black frames
compressed away. So this module collects *every* cheap signal at once and
leaves the judgement to :mod:`broadcastbrain.segment.fusion`.

Signals collected, all in **one decode pass**:

===================  ====================================================
``black``            ``blackdetect`` -- the classic hard cut
``silence``          ``silencedetect`` -- audio drops at pod boundaries
``scene``            ``scdet`` -- visual discontinuity
``freeze``           ``freezedetect`` -- slates, test cards, TV Guide pages
``aspect_change``    ``cropdetect`` -- 4:3 show into a 16:9 spot
``loudness_jump``    ``ebur128`` -- ads were mixed hot in this era
``logo_change``      ``scdet`` on a cropped corner -- the network bug
                     disappearing is one of the strongest commercial cues
                     available before you have run any AI at all
===================  ====================================================

One decode pass matters: at 100 TB, decoding the same six-hour capture six
times is the difference between a night and a week.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..config import Settings
from ..constants import FFMPEG_SIGNAL_TIMEOUT_S, LOGO_REGIONS
from ..errors import SubprocessError
from ..media.probe import MediaInfo
from ..types import Signal, SignalKind
from ..util.proc import run

log = logging.getLogger(__name__)

# --- stderr patterns ---------------------------------------------------------
_RE_BLACK = re.compile(
    r"black_start:(?P<start>[\d.]+)\s+black_end:(?P<end>[\d.]+)\s+black_duration:(?P<dur>[\d.]+)"
)
_RE_SILENCE_START = re.compile(r"silence_start:\s*(?P<start>-?[\d.]+)")
_RE_SILENCE_END = re.compile(
    r"silence_end:\s*(?P<end>[\d.]+)\s*\|\s*silence_duration:\s*(?P<dur>[\d.]+)"
)
_RE_FREEZE_START = re.compile(r"freeze_start:\s*(?P<start>[\d.]+)")
_RE_FREEZE_END = re.compile(r"freeze_end:\s*(?P<end>[\d.]+)")
_RE_CROP = re.compile(r"t:(?P<t>[\d.]+).*?crop=(?P<w>\d+):(?P<h>\d+):(?P<x>\d+):(?P<y>\d+)")
_RE_EBUR128 = re.compile(r"t:\s*(?P<t>[\d.]+)\s+TARGET:[^M]*M:\s*(?P<m>-?[\d.inf]+)")

# --- metadata-print patterns (scdet branches write to their own files) -------
_RE_META_FRAME = re.compile(r"^frame:\d+\s+pts:\S+\s+pts_time:(?P<t>[\d.]+)")
_RE_META_SCORE = re.compile(r"^lavfi\.scd\.score=(?P<score>[\d.]+)")


@dataclass
class SignalReport:
    """Everything the collector learned about one source."""

    duration_s: float
    signals: list[Signal] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    degraded: bool = False
    note: str = ""

    def by_kind(self, kind: SignalKind) -> list[Signal]:
        return [s for s in self.signals if s.kind is kind]

    def summary(self) -> dict[str, object]:
        return {
            "duration_s": round(self.duration_s, 3),
            "total_signals": len(self.signals),
            "counts": dict(self.counts),
            "degraded": self.degraded,
            "note": self.note,
        }


class SignalCollector:
    """Runs the detector pass and turns ffmpeg's chatter into typed signals."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.seg = settings.segmentation

    # -- public API ---------------------------------------------------------

    def collect(
        self,
        source: Path,
        info: MediaInfo,
        *,
        progress: Optional[Callable[[float, str], None]] = None,
        timeout_s: float = FFMPEG_SIGNAL_TIMEOUT_S,
    ) -> SignalReport:
        """Collect every enabled signal from ``source``.

        Falls back to a minimal black+silence graph if the full filter graph is
        rejected (older ffmpeg builds lack ``scdet`` or ``freezedetect``). A
        degraded report is far better than a failed source.
        """
        duration = info.duration_s or 0.0
        with tempfile.TemporaryDirectory(prefix="bb-signals-") as tmpdir:
            work = Path(tmpdir)
            scene_file = work / "scene.txt"
            logo_file = work / "logo.txt"
            try:
                report = self._run_pass(
                    source, info, duration, scene_file, logo_file, progress, timeout_s, full=True
                )
            except SubprocessError as exc:
                log.warning(
                    "full detector graph failed for %s (%s); retrying with the minimal graph",
                    source.name, str(exc)[:200],
                )
                report = self._run_pass(
                    source, info, duration, scene_file, logo_file, progress, timeout_s, full=False
                )
                report.degraded = True
                report.note = "minimal detector graph (ffmpeg rejected the full one)"
        report.signals.sort(key=lambda s: s.time_s)
        report.counts = _count_kinds(report.signals)
        log.info(
            "%s: %d signals collected %s",
            source.name, len(report.signals), report.counts,
        )
        return report

    # -- internals ----------------------------------------------------------

    def _run_pass(
        self,
        source: Path,
        info: MediaInfo,
        duration: float,
        scene_file: Path,
        logo_file: Path,
        progress: Optional[Callable[[float, str], None]],
        timeout_s: float,
        *,
        full: bool,
    ) -> SignalReport:
        report = SignalReport(duration_s=duration)
        state = _ParserState()
        handle_line = self._make_line_handler(report, state, duration, progress)

        args = self._build_command(source, info, scene_file, logo_file, full=full)
        run(
            args,
            timeout_s=timeout_s,
            nice=self.settings.workers.subprocess_nice,
            check=True,
            stderr_tail_lines=0,
            on_line=handle_line,
            log_command=self.settings.logging.log_subprocess_commands,
        )

        state.flush(report)
        if full:
            report.signals.extend(
                _parse_metadata_file(scene_file, SignalKind.SCENE, self.seg.scene_threshold)
            )
            report.signals.extend(
                _parse_metadata_file(
                    logo_file, SignalKind.LOGO_CHANGE, self.seg.logo_scene_threshold
                )
            )
        return report

    def _build_command(
        self, source: Path, info: MediaInfo, scene_file: Path, logo_file: Path, *, full: bool
    ) -> list[str]:
        seg = self.seg
        has_audio = info.has_audio
        has_video = info.has_video

        video_chains: list[str] = []
        maps: list[str] = []

        if has_video:
            # Detectors run on a downscaled, frame-dropped copy: an eighth of
            # the pixels is plenty to see a fade to black and costs a fraction
            # of the CPU.
            pre = f"scale={seg.analysis_width}:-2:flags=fast_bilinear,fps={seg.analysis_fps}"
            main_filters = [
                pre,
                f"blackdetect=d={seg.black_min_duration_s}"
                f":pic_th={seg.black_picture_ratio}:pix_th={seg.black_pixel_threshold}",
            ]
            if full:
                main_filters.append(
                    f"freezedetect=n={seg.freeze_noise_db}dB:d={seg.freeze_min_duration_s}"
                )
                if seg.crop_enabled:
                    # reset_count keeps cropdetect responsive to real changes
                    # instead of latching onto the first frame's borders.
                    main_filters.append("cropdetect=limit=24:round=2:reset_count=120")
                main_filters.append(f"scdet=threshold={seg.scene_threshold}")
                main_filters.append(
                    f"metadata=mode=print:key=lavfi.scd.score:file={_escape(scene_file)}"
                )

            if full and seg.logo_enabled:
                region = LOGO_REGIONS.get(seg.logo_corner, LOGO_REGIONS["top_right"])
                x, y, w, h = region
                crop = f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}"
                video_chains.append("[0:v]split=2[sig_main][sig_logo]")
                video_chains.append("[sig_main]" + ",".join(main_filters) + "[vout]")
                video_chains.append(
                    f"[sig_logo]{crop},{pre},scdet=threshold={seg.logo_scene_threshold},"
                    f"metadata=mode=print:key=lavfi.scd.score:file={_escape(logo_file)}[lout]"
                )
                maps += ["-map", "[vout]", "-map", "[lout]"]
            else:
                video_chains.append("[0:v]" + ",".join(main_filters) + "[vout]")
                maps += ["-map", "[vout]"]

        if has_audio:
            audio_filters = [
                f"silencedetect=n={seg.silence_noise_db}dB:d={seg.silence_min_duration_s}"
            ]
            if full and seg.loudness_enabled:
                audio_filters.append("ebur128=metadata=1:framelog=verbose")
            video_chains.append("[0:a]" + ",".join(audio_filters) + "[aout]")
            maps += ["-map", "[aout]"]

        # Output is discarded -- only the detectors' log output matters -- so
        # no encoder options are needed at all.
        return [
            self.settings.ffmpeg_binary,
            "-hide_banner",
            "-nostdin",
            "-loglevel", "info",  # detectors report at INFO
            "-i", str(source),
            "-filter_complex", ";".join(video_chains),
            *maps,
            "-f", "null",
            "-",
        ]

    def _make_line_handler(
        self,
        report: SignalReport,
        state: _ParserState,
        duration: float,
        progress: Optional[Callable[[float, str], None]],
    ) -> Callable[[str], None]:
        seg = self.seg

        def handle(line: str) -> None:
            if "black_start" in line:
                match = _RE_BLACK.search(line)
                if match:
                    start = float(match.group("start"))
                    end = float(match.group("end"))
                    length = float(match.group("dur"))
                    # The *middle* of a black run is the true cut point: black
                    # spans the tail of one clip and the head of the next.
                    report.signals.append(
                        Signal(
                            kind=SignalKind.BLACK,
                            time_s=(start + end) / 2.0,
                            strength=min(1.0, 0.5 + length),
                            duration_s=length,
                            detail=f"{start:.2f}-{end:.2f}",
                        )
                    )
                return

            if "silence_start" in line:
                match = _RE_SILENCE_START.search(line)
                if match:
                    state.silence_start = max(0.0, float(match.group("start")))
                return

            if "silence_end" in line:
                match = _RE_SILENCE_END.search(line)
                if match:
                    end = float(match.group("end"))
                    length = float(match.group("dur"))
                    start = state.silence_start if state.silence_start is not None else end - length
                    report.signals.append(
                        Signal(
                            kind=SignalKind.SILENCE,
                            time_s=(start + end) / 2.0,
                            strength=min(1.0, 0.4 + length / 2.0),
                            duration_s=length,
                        )
                    )
                    state.silence_start = None
                return

            if "freeze_start" in line:
                match = _RE_FREEZE_START.search(line)
                if match:
                    state.freeze_start = float(match.group("start"))
                return

            if "freeze_end" in line:
                match = _RE_FREEZE_END.search(line)
                if match:
                    end = float(match.group("end"))
                    start = state.freeze_start if state.freeze_start is not None else end
                    report.signals.append(
                        Signal(
                            kind=SignalKind.FREEZE,
                            time_s=start,
                            strength=0.8,
                            duration_s=max(0.0, end - start),
                        )
                    )
                    state.freeze_start = None
                return

            if seg.crop_enabled and "crop=" in line:
                match = _RE_CROP.search(line)
                if match:
                    geometry = (int(match.group("w")), int(match.group("h")))
                    time_s = float(match.group("t"))
                    if state.crop is not None and geometry != state.crop:
                        report.signals.append(
                            Signal(
                                kind=SignalKind.ASPECT_CHANGE,
                                time_s=time_s,
                                strength=0.9,
                                detail=f"{state.crop[0]}x{state.crop[1]} -> {geometry[0]}x{geometry[1]}",
                            )
                        )
                    state.crop = geometry
                return

            if seg.loudness_enabled and " M:" in line and line.lstrip().startswith("t:"):
                match = _RE_EBUR128.search(line)
                if match:
                    try:
                        momentary = float(match.group("m"))
                    except ValueError:
                        return
                    time_s = float(match.group("t"))
                    if state.loudness is not None:
                        delta = momentary - state.loudness
                        if abs(delta) >= seg.loudness_jump_lu:
                            report.signals.append(
                                Signal(
                                    kind=SignalKind.LOUDNESS_JUMP,
                                    time_s=time_s,
                                    strength=min(1.0, abs(delta) / 12.0),
                                    detail=f"{delta:+.1f} LU",
                                )
                            )
                    state.loudness = momentary
                    if progress and duration > 0:
                        progress(min(1.0, time_s / duration), "collecting signals")
                return

        return handle


@dataclass
class _ParserState:
    """Carries half-finished detector events between stderr lines."""

    silence_start: Optional[float] = None
    freeze_start: Optional[float] = None
    crop: Optional[tuple[int, int]] = None
    loudness: Optional[float] = None

    def flush(self, report: SignalReport) -> None:
        """Close any event still open when the stream ended."""
        if self.silence_start is not None and report.duration_s:
            report.signals.append(
                Signal(
                    kind=SignalKind.SILENCE,
                    time_s=(self.silence_start + report.duration_s) / 2.0,
                    strength=0.5,
                    duration_s=max(0.0, report.duration_s - self.silence_start),
                    detail="unterminated silence at EOF",
                )
            )
            self.silence_start = None


def _parse_metadata_file(path: Path, kind: SignalKind, threshold: float) -> list[Signal]:
    """Parse an ffmpeg ``metadata=mode=print`` file into signals.

    The format alternates a ``frame:`` header line with ``key=value`` lines,
    so a timestamp is carried forward until its score arrives.
    """
    if not path.exists():
        return []
    signals: list[Signal] = []
    current_time: Optional[float] = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                header = _RE_META_FRAME.match(line)
                if header:
                    current_time = float(header.group("t"))
                    continue
                score_match = _RE_META_SCORE.match(line)
                if score_match and current_time is not None:
                    score = float(score_match.group("score"))
                    if score >= threshold:
                        signals.append(
                            Signal(
                                kind=kind,
                                time_s=current_time,
                                strength=min(1.0, score / 100.0 + 0.35),
                                detail=f"score={score:.1f}",
                            )
                        )
    except OSError as exc:  # pragma: no cover - transient FS problem
        log.debug("could not read metadata file %s: %s", path, exc)
    return signals


def _count_kinds(signals: list[Signal]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for signal in signals:
        counts[signal.kind.value] = counts.get(signal.kind.value, 0) + 1
    return counts


def _escape(path: Path) -> str:
    r"""Escape a path for use inside an ffmpeg filter argument.

    ffmpeg's filter parser treats ``:``, ``\`` and ``'`` specially, and on
    Windows a temp path contains all three.
    """
    text = str(path)
    for char in ("\\", ":", "'"):
        text = text.replace(char, "\\" + char)
    return text
