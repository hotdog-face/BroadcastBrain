"""Content hashing: exact (SHA256) and perceptual (dHash sequences).

Exact hashing is what makes the pipeline idempotent -- a file that has already
been ingested is recognized before a single frame is decoded. Perceptual
hashing is what finds the same commercial re-encoded three different ways.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..constants import HASH_READ_CHUNK_BYTES, PHASH_BITS, QUICK_HASH_CHUNK_BYTES


def sha256_file(path: Path, *, chunk_bytes: int = HASH_READ_CHUNK_BYTES) -> str:
    """Full-content SHA256. Streams, so a 40 GB capture costs no extra RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def quick_signature(path: Path, *, chunk_bytes: int = QUICK_HASH_CHUNK_BYTES) -> str:
    """Cheap identity probe: size + head + tail.

    Used by discovery to skip files that are obviously already known before
    paying for a full SHA256 of a multi-gigabyte recording. Collisions are
    possible in theory, so a match is always confirmed with the real hash.
    """
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_bytes))
        if size > chunk_bytes * 2:
            handle.seek(-chunk_bytes, 2)
            digest.update(handle.read(chunk_bytes))
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_id(*parts: object) -> str:
    """Deterministic 32-hex identifier from arbitrary parts.

    Used for dedupe keys on queue jobs, so enqueueing the same work twice is a
    no-op rather than a duplicate run.
    """
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Perceptual hashing
# ---------------------------------------------------------------------------


def dhash_from_luma(luma: Sequence[Sequence[int]]) -> int:
    """Difference hash of a small grayscale image.

    Args:
        luma: Row-major grayscale matrix with 9 columns and 8 rows
            (``PHASH_BITS`` = 64 comparisons). Each row contributes 8 bits by
            comparing horizontally adjacent pixels, which makes the hash
            invariant to brightness shifts -- exactly what you need when the
            same spot was captured off VHS and off a DVR.

    Returns:
        A 64-bit integer hash.
    """
    bits = 0
    count = 0
    for row in luma:
        for x in range(len(row) - 1):
            bits = (bits << 1) | (1 if row[x + 1] > row[x] else 0)
            count += 1
    if count != PHASH_BITS:
        raise ValueError(f"expected {PHASH_BITS} comparisons, produced {count}")
    return bits


def hamming(a: int, b: int) -> int:
    """Population count of the XOR -- the classic perceptual hash distance."""
    return bin(to_unsigned64(a) ^ to_unsigned64(b)).count("1")


#: Perceptual hashes are unsigned 64-bit values, but SQLite's INTEGER (and
#: PostgreSQL's BIGINT) are *signed* 64-bit. A hash with the top bit set
#: overflows on insert. These two functions are the reinterpretation, applied
#: on the way into and out of the database.
_UINT64_RANGE = 1 << 64
_INT64_MAX = 1 << 63


def to_signed64(value: int) -> int:
    """Reinterpret an unsigned 64-bit value as signed, for storage."""
    value &= _UINT64_RANGE - 1
    return value - _UINT64_RANGE if value >= _INT64_MAX else value


def to_unsigned64(value: int) -> int:
    """Inverse of :func:`to_signed64`."""
    return value + _UINT64_RANGE if value < 0 else value


def encode_hash_sequence(hashes: Iterable[int]) -> bytes:
    """Pack a sequence of 64-bit hashes into a compact BLOB for the database."""
    return b"".join(int(h).to_bytes(8, "big", signed=False) for h in hashes)


def decode_hash_sequence(blob: Optional[bytes]) -> list[int]:
    """Inverse of :func:`encode_hash_sequence`."""
    if not blob:
        return []
    if len(blob) % 8:
        raise ValueError("hash sequence blob length must be a multiple of 8")
    return [int.from_bytes(blob[i : i + 8], "big", signed=False) for i in range(0, len(blob), 8)]


def sequence_distance(left: Sequence[int], right: Sequence[int]) -> float:
    """Normalized perceptual distance between two clip hash sequences.

    Sequences are sampled at the same *relative* positions in each clip, so
    they align even when the two copies differ slightly in length. Comparison
    is truncated to the shorter sequence and the leftover frames are charged a
    penalty, which stops a 5-second bumper from matching the first 5 seconds of
    a 30-second spot.

    Returns:
        0.0 for identical, 1.0 for maximally different.
    """
    if not left or not right:
        return 1.0
    n = min(len(left), len(right))
    total = sum(hamming(left[i], right[i]) for i in range(n))
    distance = total / (n * PHASH_BITS)
    length_penalty = abs(len(left) - len(right)) / max(len(left), len(right))
    return min(1.0, distance + length_penalty * 0.5)


def fingerprint_similarity(left: Optional[str], right: Optional[str]) -> float:
    """Similarity of two Chromaprint fingerprints, as a 0-1 score.

    Chromaprint's compressed base64 form is compared token-wise. This is a
    coarse but dependency-free comparison; when ``pyacoustid`` is installed the
    audio analyzer stores the raw integer vector too, and that path is used
    instead.
    """
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    a, b = left.strip(), right.strip()
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / max(len(a), len(b))
