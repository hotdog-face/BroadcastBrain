"""Deduplication: exact, perceptual and acoustic identity."""

from __future__ import annotations

from .hashes import (
    compare_sequences,
    compute_clip_fingerprints,
    compute_phash_sequence,
    hashes_from_raw,
    head_distance,
)
from .resolver import DedupeOutcome, Deduplicator, DuplicateMatch, duplicate_statistics

__all__ = [
    "DedupeOutcome",
    "Deduplicator",
    "DuplicateMatch",
    "compare_sequences",
    "compute_clip_fingerprints",
    "compute_phash_sequence",
    "duplicate_statistics",
    "hashes_from_raw",
    "head_distance",
]
