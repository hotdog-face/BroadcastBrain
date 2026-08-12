"""Categorization: vote aggregation, confidence, and the unknown outcome."""

from __future__ import annotations

import pytest

from broadcastbrain.categorize.rules import (
    MIN_DECISION_CONFIDENCE,
    decide,
    merge_votes,
    subcategory_for,
    tally,
)
from broadcastbrain.types import Category


class TestTally:
    def test_weights_accumulate(self) -> None:
        votes = [
            (Category.COMMERCIAL, 0.4, "brand on screen"),
            (Category.COMMERCIAL, 0.3, "call to action"),
            (Category.PROMO, 0.2, "next on"),
        ]
        totals = tally(votes)
        assert totals[Category.COMMERCIAL] == pytest.approx(0.7)
        assert totals[Category.PROMO] == pytest.approx(0.2)

    def test_zero_and_negative_weights_are_ignored(self) -> None:
        assert tally([(Category.MOVIE, 0.0, "x"), (Category.MOVIE, -1.0, "y")]) == {}


class TestDecide:
    def test_no_votes_means_unknown_not_a_guess(self) -> None:
        decision = decide([])
        assert decision.category is Category.UNKNOWN
        assert decision.confidence == 0.0

    def test_several_agreeing_votes_beat_one_strong_dissent(self) -> None:
        votes = [
            (Category.COMMERCIAL, 0.45, "duration matches the 30s ad slot"),
            (Category.COMMERCIAL, 0.4, "brand on screen"),
            (Category.COMMERCIAL, 0.35, "no network bug"),
            (Category.PROMO, 0.55, "next on"),
        ]
        assert decide(votes).category is Category.COMMERCIAL

    def test_weak_evidence_falls_back_to_unknown(self) -> None:
        """A single feeble vote must not produce a confident label."""
        decision = decide([(Category.MOVIE, 0.12, "letterboxed")])
        assert decision.confidence < MIN_DECISION_CONFIDENCE
        assert decision.category is Category.UNKNOWN

    def test_unanimous_strong_evidence_is_confident(self) -> None:
        votes = [
            (Category.COMMERCIAL, 0.6, "spoken: call now"),
            (Category.COMMERCIAL, 0.55, "on-screen text: 1-800"),
            (Category.COMMERCIAL, 0.45, "duration matches the 30s ad slot"),
        ]
        decision = decide(votes)
        assert decision.category is Category.COMMERCIAL
        assert decision.confidence > 0.6

    def test_contested_evidence_lowers_confidence(self) -> None:
        clear = decide([(Category.COMMERCIAL, 0.8, "a"), (Category.COMMERCIAL, 0.8, "b")])
        contested = decide([(Category.COMMERCIAL, 0.8, "a"), (Category.PROMO, 0.75, "b")])
        assert contested.confidence < clear.confidence

    def test_evidence_is_retained_for_explanation(self) -> None:
        decision = decide([(Category.PROMO, 0.5, "speech: next on")])
        assert len(decision.evidence) == 1
        assert decision.evidence[0].reason == "speech: next on"
        assert decision.as_dict()["evidence"][0]["category"] == "promo"

    def test_runner_up_is_reported(self) -> None:
        decision = decide(
            [(Category.COMMERCIAL, 0.9, "a"), (Category.PROMO, 0.6, "b")]
        )
        assert decision.runner_up is Category.PROMO
        assert decision.runner_up_confidence > 0

    def test_operator_lock_wins_outright(self) -> None:
        """A human decision is never overruled by automation."""
        decision = decide(
            [(Category.COMMERCIAL, 5.0, "overwhelming")], locked_category=Category.MUSIC_VIDEO
        )
        assert decision.category is Category.MUSIC_VIDEO
        assert decision.confidence == 1.0

    def test_flags_follow_the_category(self) -> None:
        flags = decide([(Category.COMMERCIAL, 1.5, "x")]).flags()
        assert flags["is_commercial"] is True
        assert flags["is_promo"] is False


class TestSubcategory:
    @pytest.mark.parametrize("duration,expected", [(15.0, "15s"), (30.0, "30s"), (60.0, "60s")])
    def test_ad_slot_lengths(self, duration: float, expected: str) -> None:
        decision = decide([(Category.COMMERCIAL, 1.5, "x")])
        assert subcategory_for(decision, duration) == expected

    def test_odd_length_commercial(self) -> None:
        decision = decide([(Category.COMMERCIAL, 1.5, "x")])
        assert subcategory_for(decision, 37.0) == "other-length"

    def test_episode_lengths(self) -> None:
        decision = decide([(Category.TV_EPISODE, 1.5, "x")])
        assert subcategory_for(decision, 1320.0) == "half-hour"
        assert subcategory_for(decision, 2640.0) == "hour"

    def test_tags_influence_subcategory(self) -> None:
        decision = decide([(Category.MOVIE, 1.5, "x")])
        assert subcategory_for(decision, 5400.0, ["letterboxed"]) == "letterboxed"
        assert subcategory_for(decision, 5400.0, []) == "full-frame"


def test_merge_votes_concatenates() -> None:
    merged = merge_votes(
        [(Category.COMMERCIAL, 0.5, "a")], [(Category.PROMO, 0.4, "b")]
    )
    assert len(merged) == 2
