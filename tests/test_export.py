"""Fieldstation export: pod construction and schedule shape.

The brief's example is the specification here:

    MythBusters -> commercial -> Discovery promo -> commercial -> Dirty Jobs
"""

from __future__ import annotations

import pytest

from broadcastbrain.export.scheduling import (
    ExportClip,
    InterstitialPool,
    SchedulingConfig,
    build_block,
    build_pod,
    build_schedule,
    schedule_statistics,
    suggest_channels,
)
from broadcastbrain.types import Category


def clip(
    index: int, duration: float, category: Category, **kwargs: object
) -> ExportClip:
    return ExportClip(
        id=f"clip-{index}",
        path=f"/library/{index}.mkv",
        duration_s=duration,
        category=category,
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture()
def config() -> SchedulingConfig:
    return SchedulingConfig(break_interval_s=480.0, pod_target_s=120.0, seed=7)


@pytest.fixture()
def interstitials() -> list[ExportClip]:
    return (
        [clip(100 + i, 30.0, Category.COMMERCIAL, title=f"Spot {i}") for i in range(12)]
        + [
            clip(200 + i, 15.0, Category.PROMO, title=f"Promo {i}", network="Discovery Channel")
            for i in range(4)
        ]
        + [
            clip(300 + i, 5.0, Category.NETWORK_BUMPER, title=f"Bumper {i}",
                 network="Discovery Channel")
            for i in range(3)
        ]
    )


@pytest.fixture()
def programs() -> list[ExportClip]:
    return [
        clip(1, 2640.0, Category.TV_EPISODE, title="MythBusters", series="MythBusters",
             network="Discovery Channel"),
        clip(2, 1320.0, Category.TV_EPISODE, title="Dirty Jobs", series="Dirty Jobs",
             network="Discovery Channel"),
    ]


class TestPods:
    def test_pod_reaches_roughly_the_target_length(
        self, interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        pool = InterstitialPool(interstitials, seed=config.seed)
        pod = build_pod(pool, config, position_s=600.0)
        assert config.pod_target_s * 0.7 <= pod.duration_s <= config.pod_target_s * 1.4

    def test_pod_contains_a_promo(
        self, interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        pool = InterstitialPool(interstitials, seed=config.seed)
        pod = build_pod(pool, config, position_s=600.0, network="Discovery Channel")
        assert any(c.category is Category.PROMO for c in pod.clips)

    def test_pod_respects_the_spot_ceiling(
        self, interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        pool = InterstitialPool(interstitials, seed=config.seed)
        pod = build_pod(pool, config, position_s=600.0)
        spots = [c for c in pod.clips if c.category is Category.COMMERCIAL]
        assert len(spots) <= config.pod_max_spots

    def test_pod_is_empty_when_there_is_no_inventory(self, config: SchedulingConfig) -> None:
        pod = build_pod(InterstitialPool([], seed=1), config, position_s=100.0)
        assert pod.clips == []

    def test_spots_are_not_repeated_within_a_pod(
        self, interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        pool = InterstitialPool(interstitials, seed=config.seed)
        pod = build_pod(pool, config, position_s=600.0)
        assert len({c.id for c in pod.clips}) == len(pod.clips)


class TestBlocks:
    def test_long_program_gets_several_breaks(
        self, programs: list[ExportClip], interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        pool = InterstitialPool(interstitials, seed=config.seed)
        block = build_block(programs[0], pool, config)
        assert len(block.pods) >= 4  # a 44-minute show at 8-minute intervals

    def test_breaks_are_inside_the_program(
        self, programs: list[ExportClip], interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        pool = InterstitialPool(interstitials, seed=config.seed)
        block = build_block(programs[0], pool, config)
        assert all(0 < pod.position_s < programs[0].duration_s for pod in block.pods)

    def test_short_program_gets_a_trailing_break_only(
        self, interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        short = clip(9, 120.0, Category.TV_EPISODE, title="Short", network="Discovery Channel")
        pool = InterstitialPool(interstitials, seed=config.seed)
        block = build_block(short, pool, config)
        assert len(block.pods) == 1
        assert block.pods[0].position_s == pytest.approx(short.duration_s)

    def test_block_opens_with_a_bumper(
        self, programs: list[ExportClip], interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        pool = InterstitialPool(interstitials, seed=config.seed)
        block = build_block(programs[0], pool, config)
        assert block.lead_in and block.lead_in[0].category is Category.NETWORK_BUMPER


class TestSchedule:
    def test_the_brief_example_is_produced(
        self, programs: list[ExportClip], interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        """MythBusters -> commercial -> Discovery promo -> commercial -> Dirty Jobs."""
        blocks = build_schedule(programs, interstitials, config)
        assert [b.program.title for b in blocks] == ["MythBusters", "Dirty Jobs"]

        first_pod = blocks[0].pods[0]
        kinds = [c.category for c in first_pod.clips]
        assert kinds[0] is Category.COMMERCIAL
        assert Category.PROMO in kinds
        assert kinds.count(Category.COMMERCIAL) >= 2

    def test_schedule_is_deterministic(
        self, programs: list[ExportClip], interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        first = build_schedule(programs, interstitials, config)
        second = build_schedule(programs, interstitials, config)
        assert [b.as_dict() for b in first] == [b.as_dict() for b in second]

    def test_different_seeds_give_different_schedules(
        self, programs: list[ExportClip], interstitials: list[ExportClip]
    ) -> None:
        first = build_schedule(programs, interstitials, SchedulingConfig(seed=1))
        second = build_schedule(programs, interstitials, SchedulingConfig(seed=99))
        assert [b.as_dict() for b in first] != [b.as_dict() for b in second]

    def test_statistics(
        self, programs: list[ExportClip], interstitials: list[ExportClip], config: SchedulingConfig
    ) -> None:
        stats = schedule_statistics(build_schedule(programs, interstitials, config))
        assert stats["blocks"] == 2
        assert stats["pods"] > 0
        assert stats["total_duration_s"] > sum(p.duration_s for p in programs)


class TestChannelSuggestions:
    def test_a_network_with_both_kinds_is_ready(
        self, programs: list[ExportClip], interstitials: list[ExportClip]
    ) -> None:
        suggestions = suggest_channels(programs + interstitials)
        discovery = next(s for s in suggestions if s["network"] == "Discovery Channel")
        assert discovery["ready"] is True

    def test_programming_without_interstitials_is_flagged(
        self, programs: list[ExportClip]
    ) -> None:
        suggestion = suggest_channels(programs)[0]
        assert suggestion["ready"] is False
        assert "no commercials" in str(suggestion["notes"])

    def test_unaffiliated_clips_are_grouped(self, interstitials: list[ExportClip]) -> None:
        names = [s["network"] for s in suggest_channels(interstitials)]
        assert "Unaffiliated" in names
