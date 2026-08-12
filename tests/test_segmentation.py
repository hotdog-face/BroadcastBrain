"""Segmentation: clustering, scoring, the ad grid, and segment repair.

These are the tests that matter most for output quality -- everything
downstream is only as good as where the cuts landed -- and they run entirely on
lists of floats, with no ffmpeg and no media.
"""

from __future__ import annotations

import pytest

from broadcastbrain.constants import SIGNAL_WEIGHTS
from broadcastbrain.segment.fusion import (
    FusionConfig,
    boundaries_to_segments,
    cluster_signals,
    fuse,
    grid_alignment_bonus,
    merge_short,
    select_boundaries,
    split_oversized,
)
from broadcastbrain.types import Signal, SignalKind


@pytest.fixture()
def config() -> FusionConfig:
    return FusionConfig(
        cluster_window_s=1.2,
        min_boundary_score=1.05,
        min_segment_s=3.0,
        max_segment_s=600.0,
        weights=dict(SIGNAL_WEIGHTS),
        grid_bonus=0.25,
    )


def signal(kind: SignalKind, t: float, strength: float = 1.0) -> Signal:
    return Signal(kind=kind, time_s=t, strength=strength)


class TestClustering:
    def test_nearby_signals_collapse_into_one_candidate(self, config: FusionConfig) -> None:
        signals = [
            signal(SignalKind.BLACK, 100.0),
            signal(SignalKind.SILENCE, 100.4),
            signal(SignalKind.SCENE, 100.8),
        ]
        candidates = cluster_signals(signals, config.cluster_window_s)
        assert len(candidates) == 1
        assert set(candidates[0].kinds) == {
            SignalKind.BLACK, SignalKind.SILENCE, SignalKind.SCENE
        }

    def test_distant_signals_stay_separate(self, config: FusionConfig) -> None:
        signals = [signal(SignalKind.BLACK, 100.0), signal(SignalKind.BLACK, 140.0)]
        assert len(cluster_signals(signals, config.cluster_window_s)) == 2

    def test_repeated_kind_does_not_inflate_the_score(self, config: FusionConfig) -> None:
        """Three scene hits are one opinion, not three."""
        repeated = cluster_signals(
            [signal(SignalKind.SCENE, 50.0 + i * 0.2) for i in range(3)], config.cluster_window_s
        )
        diverse = cluster_signals(
            [
                signal(SignalKind.SCENE, 50.0),
                signal(SignalKind.BLACK, 50.2),
                signal(SignalKind.SILENCE, 50.4),
            ],
            config.cluster_window_s,
        )
        assert diverse[0].score(config.weights) > repeated[0].score(config.weights)

    def test_empty_input(self, config: FusionConfig) -> None:
        assert cluster_signals([], config.cluster_window_s) == []


class TestGridBonus:
    def test_exact_grid_hit_gets_the_full_bonus(self, config: FusionConfig) -> None:
        assert grid_alignment_bonus(30.0, 0.0, config) == pytest.approx(config.grid_bonus)

    def test_off_grid_gets_nothing(self, config: FusionConfig) -> None:
        assert grid_alignment_bonus(37.0, 0.0, config) == 0.0

    def test_bonus_tapers_with_distance(self, config: FusionConfig) -> None:
        close = grid_alignment_bonus(30.2, 0.0, config)
        far = grid_alignment_bonus(30.8, 0.0, config)
        assert close > far > 0.0

    def test_measured_from_the_previous_boundary(self, config: FusionConfig) -> None:
        # 630 is not a grid multiple of 0, but it is 30s after 600.
        assert grid_alignment_bonus(630.0, 600.0, config) > 0.0


class TestBoundarySelection:
    def test_weak_lone_signal_is_rejected(self, config: FusionConfig) -> None:
        candidates = cluster_signals([signal(SignalKind.SCENE, 50.0, 0.4)], config.cluster_window_s)
        assert select_boundaries(candidates, 600.0, config) == []

    def test_black_plus_silence_is_accepted(self, config: FusionConfig) -> None:
        candidates = cluster_signals(
            [signal(SignalKind.BLACK, 50.0), signal(SignalKind.SILENCE, 50.2)],
            config.cluster_window_s,
        )
        boundaries = select_boundaries(candidates, 600.0, config)
        assert len(boundaries) == 1
        assert boundaries[0].is_strong

    def test_minimum_segment_length_is_enforced(self, config: FusionConfig) -> None:
        signals = []
        for t in (100.0, 101.0, 102.0):  # all within min_segment_s of each other
            signals += [signal(SignalKind.BLACK, t), signal(SignalKind.SILENCE, t)]
        candidates = cluster_signals(signals, 0.1)
        boundaries = select_boundaries(candidates, 600.0, config)
        assert len(boundaries) == 1

    def test_boundaries_outside_the_source_are_dropped(self, config: FusionConfig) -> None:
        candidates = cluster_signals(
            [signal(SignalKind.BLACK, 900.0), signal(SignalKind.SILENCE, 900.1)], 1.2
        )
        assert select_boundaries(candidates, 600.0, config) == []


class TestSegments:
    def test_segments_cover_the_whole_source(self, config: FusionConfig) -> None:
        signals = []
        for t in (240.0, 270.0, 300.0):
            signals += [signal(SignalKind.BLACK, t), signal(SignalKind.SILENCE, t + 0.2)]
        segments, _ = fuse(signals, 600.0, config)

        assert segments[0].start_s == 0.0
        assert segments[-1].end_s == pytest.approx(600.0)
        for earlier, later in zip(segments, segments[1:]):
            assert earlier.end_s == pytest.approx(later.start_s), "gap between segments"

    def test_commercial_pod_is_recovered(self, config: FusionConfig) -> None:
        """The headline case: four 30-second spots between two programs."""
        signals = []
        for t in (600.0, 630.0, 660.0, 690.0, 720.0):
            signals += [signal(SignalKind.BLACK, t, 0.9), signal(SignalKind.SILENCE, t + 0.1, 0.8)]
        segments, boundaries = fuse(signals, 1200.0, config)

        assert len(boundaries) == 5
        spots = [s for s in segments if 28.0 <= s.duration_s <= 32.0]
        assert len(spots) == 4

    def test_short_fragments_are_merged_away(self, config: FusionConfig) -> None:
        from broadcastbrain.types import Segment

        segments = [
            Segment(index=0, start_s=0.0, end_s=100.0, end_score=2.0),
            Segment(index=1, start_s=100.0, end_s=101.0, start_score=2.0, end_score=0.2),
            Segment(index=2, start_s=101.0, end_s=200.0, start_score=0.2),
        ]
        merged = merge_short(segments, config)
        assert all(s.duration_s >= config.min_segment_s for s in merged)
        assert merged[-1].end_s == pytest.approx(200.0)

    def test_oversized_segment_is_resplit(self, config: FusionConfig) -> None:
        from broadcastbrain.types import Segment

        # One weak candidate inside a segment that is far too long.
        candidates = cluster_signals(
            [signal(SignalKind.BLACK, 400.0, 0.7), signal(SignalKind.SCENE, 400.3, 0.6)], 1.2
        )
        oversized = [Segment(index=0, start_s=0.0, end_s=900.0)]
        result = split_oversized(oversized, candidates, config)
        assert len(result) == 2
        assert result[0].end_s == pytest.approx(400.0, abs=1.0)

    def test_oversized_segment_without_candidates_is_kept_whole(self, config: FusionConfig) -> None:
        from broadcastbrain.types import Segment

        result = split_oversized([Segment(index=0, start_s=0.0, end_s=900.0)], [], config)
        assert len(result) == 1, "a clip is never discarded just because it is long"

    def test_no_signals_produces_one_whole_segment(self, config: FusionConfig) -> None:
        segments = boundaries_to_segments([], 600.0)
        assert len(segments) == 1
        assert segments[0].duration_s == pytest.approx(600.0)

    def test_zero_duration_is_handled(self, config: FusionConfig) -> None:
        assert fuse([signal(SignalKind.BLACK, 1.0)], 0.0, config) == ([], [])
