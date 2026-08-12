"""Clip fingerprints: exact, perceptual and acoustic.

The perceptual hash is computed without OpenCV, NumPy or Pillow. ffmpeg is
asked to decode the clip straight down to 9x8 grayscale raw frames -- 72 bytes
each -- which is exactly the input a difference hash needs. That keeps
deduplication working on a bare machine, and it is dramatically faster than
decoding full frames and downscaling them in Python.

Sampling is *proportional*: frames are taken at fixed fractions of the clip's
length rather than at fixed times. Two copies of the same spot that differ by
half a second of leading black still line up, which is what makes near
duplicate detection work across sources.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from ..config import Settings
from ..constants import PHASH_FRAME_COUNT
from ..errors import SubprocessError
from ..util.hashing import (
    dhash_from_luma,
    encode_hash_sequence,
    hamming,
    sequence_distance,
    sha256_file,
    to_signed64,
)
from ..util.proc import run

log = logging.getLogger(__name__)

#: dHash needs a 9x8 grayscale image: 8 rows of 9 pixels give 64 comparisons.
HASH_WIDTH = 9
HASH_HEIGHT = 8
FRAME_BYTES = HASH_WIDTH * HASH_HEIGHT


def compute_phash_sequence(
    settings: Settings,
    path: Path,
    duration_s: float,
    *,
    frame_count: int = PHASH_FRAME_COUNT,
    timeout_s: float = 900,
) -> list[int]:
    """Perceptual hash sequence for a clip.

    Returns:
        Up to ``frame_count`` 64-bit hashes, ordered by time. Empty when the
        clip could not be decoded -- callers treat that as "no perceptual
        identity", never as an error.
    """
    if duration_s <= 0 or frame_count <= 0:
        return []

    # Sample slightly inside the clip: the first and last frames are often
    # black or a dissolve, which hash identically for unrelated clips.
    fps = frame_count / max(duration_s * 0.94, 0.5)

    with tempfile.TemporaryDirectory(prefix="bb-phash-") as tmpdir:
        raw_path = Path(tmpdir) / "frames.gray"
        args = [
            settings.ffmpeg_binary,
            "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{min(0.3, duration_s * 0.03):.3f}",
            "-i", str(path),
            "-vf", f"fps={fps:.6f},scale={HASH_WIDTH}:{HASH_HEIGHT}:flags=area,format=gray",
            "-frames:v", str(frame_count),
            "-f", "rawvideo",
            str(raw_path),
        ]
        try:
            run(args, timeout_s=timeout_s, check=True, stderr_tail_lines=20,
                nice=settings.workers.subprocess_nice)
        except SubprocessError as exc:
            log.debug("perceptual hashing failed for %s: %s", path.name, exc)
            return []

        try:
            data = raw_path.read_bytes()
        except OSError:
            return []

    return hashes_from_raw(data)


def hashes_from_raw(data: bytes) -> list[int]:
    """Convert a raw 9x8 grayscale frame stream into dHash values."""
    hashes: list[int] = []
    for offset in range(0, len(data) - FRAME_BYTES + 1, FRAME_BYTES):
        chunk = data[offset : offset + FRAME_BYTES]
        rows = [
            list(chunk[row * HASH_WIDTH : (row + 1) * HASH_WIDTH]) for row in range(HASH_HEIGHT)
        ]
        try:
            hashes.append(dhash_from_luma(rows))
        except ValueError:  # pragma: no cover - malformed tail
            break
    return hashes


def compute_clip_fingerprints(
    settings: Settings,
    path: Path,
    duration_s: float,
    *,
    frame_count: int = PHASH_FRAME_COUNT,
) -> dict[str, object]:
    """Everything needed to identify a clip: SHA256 plus the perceptual hashes.

    The Chromaprint fingerprint is produced by the audio analyzer instead of
    here, because it needs the extracted analysis audio that the analysis stage
    has already paid for.
    """
    hashes = compute_phash_sequence(settings, path, duration_s, frame_count=frame_count)
    return {
        "sha256": sha256_file(path),
        "phash": encode_hash_sequence(hashes),
        # Stored signed: an unsigned 64-bit hash overflows BIGINT.
        "phash_head": to_signed64(hashes[0]) if hashes else None,
        "phash_frames": len(hashes),
    }


def compare_sequences(left: Sequence[int], right: Sequence[int]) -> float:
    """Normalized perceptual distance, 0.0 identical to 1.0 unrelated."""
    return sequence_distance(left, right)


def head_distance(left: Optional[int], right: Optional[int]) -> Optional[int]:
    """Hamming distance between two clips' first frame hashes.

    Used as a cheap pre-filter: if the opening frames are wildly different the
    full sequence comparison is not worth doing.
    """
    if left is None or right is None:
        return None
    return hamming(int(left), int(right))
