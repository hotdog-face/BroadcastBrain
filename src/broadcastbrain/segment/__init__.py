"""Multi-signal segmentation: signals -> boundaries -> segments."""

from __future__ import annotations

from .fusion import (
    Candidate,
    FusionConfig,
    boundaries_to_segments,
    cluster_signals,
    describe,
    fuse,
    grid_alignment_bonus,
    merge_short,
    select_boundaries,
    split_oversized,
)
from .segmenter import SegmentationResult, Segmenter
from .signals import SignalCollector, SignalReport

__all__ = [
    "Candidate",
    "FusionConfig",
    "SegmentationResult",
    "Segmenter",
    "SignalCollector",
    "SignalReport",
    "boundaries_to_segments",
    "cluster_signals",
    "describe",
    "fuse",
    "grid_alignment_bonus",
    "merge_short",
    "select_boundaries",
    "split_oversized",
]
