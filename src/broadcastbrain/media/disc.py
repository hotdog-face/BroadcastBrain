"""Disc images and DVD rips.

An ``.iso`` or a ``VIDEO_TS`` folder is not one recording -- it is several
titles, often with the same episode duplicated across VOB files. Feeding the
raw image into segmentation would produce garbage, so discs are expanded into
child sources first, each of which then flows through the normal pipeline.

Expansion is deliberately mount-free: mounting requires root on Linux and user
approval on macOS, neither of which belongs in an unattended overnight run.
ffmpeg reads VOB concatenations directly, and ``7z`` can list/extract an ISO
without touching the loopback device.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..constants import DISC_IMAGE_EXTENSIONS, DVD_MARKER_DIRS, MIN_SOURCE_BYTES
from ..errors import UnsupportedMediaError
from ..util.proc import run, which

log = logging.getLogger(__name__)

#: DVD VOBs are named VTS_<title>_<part>.VOB; part 0 is menu data, not video.
_VOB_PATTERN = re.compile(r"VTS_(\d{2})_(\d)\.VOB", re.IGNORECASE)


@dataclass(frozen=True)
class DiscTitle:
    """One playable title found on a disc or in a rip."""

    number: int
    parts: tuple[Path, ...]
    size_bytes: int

    @property
    def primary(self) -> Path:
        return self.parts[0]


def is_disc_image(path: Path) -> bool:
    return path.suffix.lower() in DISC_IMAGE_EXTENSIONS


def is_dvd_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any((path / marker).is_dir() for marker in DVD_MARKER_DIRS) or path.name.upper() == "VIDEO_TS"


def find_video_ts(root: Path) -> Optional[Path]:
    """Locate the ``VIDEO_TS`` directory in a rip, wherever it was nested."""
    if root.name.upper() == "VIDEO_TS":
        return root
    for marker in DVD_MARKER_DIRS:
        candidate = root / marker
        if candidate.is_dir():
            return candidate
    for candidate in root.rglob("VIDEO_TS"):
        if candidate.is_dir():
            return candidate
    return None


def titles_from_video_ts(video_ts: Path, *, min_bytes: int = MIN_SOURCE_BYTES) -> list[DiscTitle]:
    """Group VOB files into titles, skipping menus and stubs.

    Parts of a title are concatenated in order by ffmpeg's ``concat`` protocol,
    which is how a 45-minute episode split across three 1 GB VOBs becomes one
    continuous source.
    """
    grouped: dict[int, list[tuple[int, Path]]] = {}
    for entry in sorted(video_ts.iterdir()):
        if not entry.is_file():
            continue
        match = _VOB_PATTERN.fullmatch(entry.name)
        if not match:
            continue
        title_number, part_number = int(match.group(1)), int(match.group(2))
        if part_number == 0:
            continue  # menu / navigation VOB
        try:
            if entry.stat().st_size < min_bytes:
                continue
        except OSError:
            continue
        grouped.setdefault(title_number, []).append((part_number, entry))

    titles: list[DiscTitle] = []
    for number, parts in sorted(grouped.items()):
        ordered = tuple(path for _part, path in sorted(parts))
        size = sum(p.stat().st_size for p in ordered)
        titles.append(DiscTitle(number=number, parts=ordered, size_bytes=size))
    return titles


def concat_input(title: DiscTitle) -> str:
    """ffmpeg input string for a multi-part title.

    ``concat:a.VOB|b.VOB`` works for MPEG program streams specifically, which
    is exactly what DVD VOBs are.
    """
    if len(title.parts) == 1:
        return str(title.primary)
    return "concat:" + "|".join(str(p) for p in title.parts)


def extract_iso(image: Path, destination: Path, *, timeout_s: float = 7200) -> Optional[Path]:
    """Extract an ISO's ``VIDEO_TS`` without mounting it.

    Returns the extracted ``VIDEO_TS`` directory, or ``None`` when no
    extraction tool is available -- in which case the caller falls back to
    letting ffmpeg read the image directly, which works for many rips.
    """
    seven_zip = which("7z") or which("7za") or which("7zz")
    if seven_zip is None:
        log.info("7z not installed; will let ffmpeg read %s directly", image.name)
        return None

    destination.mkdir(parents=True, exist_ok=True)
    log.info("extracting %s -> %s", image.name, destination)
    run(
        [seven_zip, "x", "-y", f"-o{destination}", str(image), "VIDEO_TS/*"],
        timeout_s=timeout_s,
        check=False,
        stderr_tail_lines=40,
    )
    video_ts = find_video_ts(destination)
    if video_ts is None:
        log.warning("no VIDEO_TS found inside %s after extraction", image.name)
    return video_ts


def plan_disc_expansion(
    path: Path, work_dir: Path, *, min_title_bytes: int = 50 * 1024 * 1024
) -> list[DiscTitle]:
    """Work out the playable titles for a disc image or rip directory.

    Args:
        work_dir: Scratch space for ISO extraction.
        min_title_bytes: Titles smaller than this are almost always menus,
            previews or FBI warnings. They are skipped as *sources* -- if they
            contain real content they will still be found when the disc's main
            titles are segmented.

    Raises:
        UnsupportedMediaError: nothing playable could be identified.
    """
    video_ts: Optional[Path] = None

    if is_dvd_directory(path):
        video_ts = find_video_ts(path)
    elif is_disc_image(path):
        video_ts = extract_iso(path, work_dir / f"{path.stem}_extracted")
        if video_ts is None:
            # ffmpeg can often demux the image itself; treat it as one title.
            return [DiscTitle(number=1, parts=(path,), size_bytes=path.stat().st_size)]

    if video_ts is None:
        raise UnsupportedMediaError(f"no DVD structure found in {path}")

    titles = [t for t in titles_from_video_ts(video_ts) if t.size_bytes >= min_title_bytes]
    if not titles:
        raise UnsupportedMediaError(f"no playable titles of usable size in {video_ts}")
    log.info("%s: %d title(s) found", path.name, len(titles))
    return titles
