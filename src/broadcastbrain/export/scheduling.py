"""Building broadcast-shaped schedules from a library of clips.

The goal is the feel of 2000s cable, which is mostly a matter of getting the
*interstitials* right. A run of programming with adverts bolted on the end is
not television; television is:

    MythBusters -> commercial -> Discovery promo -> commercial -> bumper ->
    Dirty Jobs

So this module builds **pods** (a run of spots that fills a break) and inserts
them into programs at plausible intervals, following the conventions of the
era: the network's own promo sits inside a national pod, a bumper plays coming
back from break, and the pod length is a round number because the affiliate
had to hit it.

Everything here is pure and deterministic given a seed, so an export is
reproducible and testable without a database or a media file in sight.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..types import INTERSTITIAL_CATEGORIES, PROGRAM_CATEGORIES, Category


@dataclass(frozen=True)
class ExportClip:
    """The minimum a scheduler needs to know about a clip."""

    id: str
    path: str
    duration_s: float
    category: Category
    title: str = ""
    network: Optional[str] = None
    series: Optional[str] = None
    air_year: Optional[int] = None
    subcategory: Optional[str] = None
    thumbnail: Optional[str] = None

    @property
    def is_interstitial(self) -> bool:
        return self.category in INTERSTITIAL_CATEGORIES

    @property
    def is_program(self) -> bool:
        return self.category in PROGRAM_CATEGORIES

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "path": self.path,
            "duration_s": round(self.duration_s, 3),
            "category": self.category.value,
            "subcategory": self.subcategory,
            "title": self.title,
            "network": self.network,
            "series": self.series,
            "air_year": self.air_year,
            "thumbnail": self.thumbnail,
        }


@dataclass
class Pod:
    """One commercial break."""

    clips: list[ExportClip] = field(default_factory=list)
    position_s: float = 0.0

    @property
    def duration_s(self) -> float:
        return sum(clip.duration_s for clip in self.clips)

    def as_dict(self) -> dict[str, object]:
        return {
            "position_s": round(self.position_s, 2),
            "duration_s": round(self.duration_s, 2),
            "clips": [clip.as_dict() for clip in self.clips],
        }


@dataclass
class ScheduleBlock:
    """A program plus the breaks scheduled inside and after it."""

    program: ExportClip
    pods: list[Pod] = field(default_factory=list)
    lead_in: list[ExportClip] = field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return (
            self.program.duration_s
            + sum(pod.duration_s for pod in self.pods)
            + sum(clip.duration_s for clip in self.lead_in)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "program": self.program.as_dict(),
            "lead_in": [clip.as_dict() for clip in self.lead_in],
            "pods": [pod.as_dict() for pod in self.pods],
            "total_duration_s": round(self.total_duration_s, 2),
            "break_count": len(self.pods),
        }


@dataclass(frozen=True)
class SchedulingConfig:
    """Scheduling knobs, decoupled from the global settings object."""

    break_interval_s: float = 480.0
    pod_target_s: float = 120.0
    pod_min_spots: int = 2
    pod_max_spots: int = 4
    return_bumper_probability: float = 0.7
    seed: int = 42


class InterstitialPool:
    """Draws interstitials without repeating one until the pool is exhausted.

    Repetition is the thing that breaks the illusion fastest. Drawing without
    replacement, and preferring spots from the same network as the programme,
    is most of what makes a generated break sound like a real one.
    """

    def __init__(self, clips: Iterable[ExportClip], *, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._by_category: dict[Category, list[ExportClip]] = {}
        for clip in clips:
            self._by_category.setdefault(clip.category, []).append(clip)
        for pool in self._by_category.values():
            self._rng.shuffle(pool)
        self._cursor: dict[Category, int] = dict.fromkeys(self._by_category, 0)

    def available(self, category: Category) -> int:
        return len(self._by_category.get(category, ()))

    def draw(
        self,
        category: Category,
        *,
        prefer_network: Optional[str] = None,
        max_duration_s: Optional[float] = None,
    ) -> Optional[ExportClip]:
        """Take the next clip of a category, honouring soft preferences."""
        pool = self._by_category.get(category)
        if not pool:
            return None

        start = self._cursor.get(category, 0)
        fallback: Optional[int] = None

        for offset in range(len(pool)):
            index = (start + offset) % len(pool)
            clip = pool[index]
            if max_duration_s is not None and clip.duration_s > max_duration_s:
                continue
            if prefer_network and clip.network and clip.network != prefer_network:
                if fallback is None:
                    fallback = index
                continue
            self._cursor[category] = (index + 1) % len(pool)
            return clip

        if fallback is not None:
            self._cursor[category] = (fallback + 1) % len(pool)
            return pool[fallback]
        return None


def build_pod(
    pool: InterstitialPool,
    config: SchedulingConfig,
    *,
    position_s: float,
    network: Optional[str] = None,
    include_promo: bool = True,
) -> Pod:
    """Assemble one commercial break.

    Shape: national spots up to the target length, with one network promo
    dropped in the middle -- which is where local affiliates and the networks
    themselves actually put them -- and a bumper on the way back.
    """
    pod = Pod(position_s=position_s)
    remaining = config.pod_target_s

    for slot in range(config.pod_max_spots):
        if len(pod.clips) >= config.pod_max_spots:
            break
        # A promo in the second slot is the classic pattern.
        if include_promo and slot == 1 and pool.available(Category.PROMO):
            promo = pool.draw(Category.PROMO, prefer_network=network, max_duration_s=remaining)
            if promo is not None:
                pod.clips.append(promo)
                remaining -= promo.duration_s
                continue

        spot = pool.draw(Category.COMMERCIAL, max_duration_s=max(remaining, 15.0))
        if spot is None:
            break
        pod.clips.append(spot)
        remaining -= spot.duration_s
        if remaining <= 5.0 and len(pod.clips) >= config.pod_min_spots:
            break

    # Coming back from the break: a bumper or station ID re-establishes the
    # channel. This is the single most era-authentic touch in the schedule.
    if pod.clips and _chance(config, position_s, config.return_bumper_probability):
        bumper = (
            pool.draw(Category.NETWORK_BUMPER, prefer_network=network)
            or pool.draw(Category.STATION_ID, prefer_network=network)
        )
        if bumper is not None:
            pod.clips.append(bumper)
    return pod


def build_block(
    program: ExportClip, pool: InterstitialPool, config: SchedulingConfig
) -> ScheduleBlock:
    """Place breaks inside one program.

    Break positions are evenly spaced rather than random: real broadcast
    breaks land on a clock, and evenly spaced ones sound right. A short
    program gets no internal break at all, only the one that follows it.
    """
    block = ScheduleBlock(program=program)

    lead_in = pool.draw(Category.NETWORK_BUMPER, prefer_network=program.network)
    if lead_in is not None:
        block.lead_in.append(lead_in)

    if program.duration_s <= config.break_interval_s:
        block.pods.append(
            build_pod(pool, config, position_s=program.duration_s, network=program.network)
        )
        return block

    break_count = max(1, int(program.duration_s // config.break_interval_s))
    spacing = program.duration_s / (break_count + 1)
    for index in range(1, break_count + 1):
        position = spacing * index
        pod = build_pod(pool, config, position_s=position, network=program.network)
        if pod.clips:
            block.pods.append(pod)
    return block


def build_schedule(
    programs: Sequence[ExportClip],
    interstitials: Sequence[ExportClip],
    config: SchedulingConfig,
) -> list[ScheduleBlock]:
    """Build a block for every program, drawing from a shared pool.

    The pool is shared across the whole schedule so a spot used in the first
    show does not immediately reappear in the second.
    """
    pool = InterstitialPool(interstitials, seed=config.seed)
    return [build_block(program, pool, config) for program in programs]


def suggest_channels(clips: Sequence[ExportClip]) -> list[dict[str, object]]:
    """Propose one Fieldstation channel per network, with its inventory.

    A network only becomes a channel suggestion if it has both programming and
    interstitials -- a channel with shows and no adverts is not the experience
    being recreated, and the report says so rather than silently emitting a
    broken channel.
    """
    by_network: dict[str, list[ExportClip]] = {}
    for clip in clips:
        by_network.setdefault(clip.network or "Unaffiliated", []).append(clip)

    suggestions: list[dict[str, object]] = []
    for network, network_clips in sorted(by_network.items()):
        programs = [c for c in network_clips if c.is_program]
        interstitials = [c for c in network_clips if c.is_interstitial]
        program_seconds = sum(c.duration_s for c in programs)
        interstitial_seconds = sum(c.duration_s for c in interstitials)

        suggestions.append(
            {
                "network": network,
                "clip_count": len(network_clips),
                "program_count": len(programs),
                "interstitial_count": len(interstitials),
                "program_hours": round(program_seconds / 3600.0, 2),
                "interstitial_minutes": round(interstitial_seconds / 60.0, 1),
                "ready": bool(programs and interstitials),
                "notes": _channel_notes(programs, interstitials),
                "series": sorted({c.series for c in programs if c.series}),
                "years": sorted({c.air_year for c in programs if c.air_year}),
            }
        )
    suggestions.sort(key=lambda s: float(s["program_hours"]), reverse=True)  # type: ignore[arg-type]
    return suggestions


def _channel_notes(programs: Sequence[ExportClip], interstitials: Sequence[ExportClip]) -> str:
    if not programs:
        return "no programming yet -- usable as an interstitial pool only"
    if not interstitials:
        return "no commercials or bumpers yet -- breaks will be silent"
    ratio = sum(c.duration_s for c in interstitials) / max(
        1.0, sum(c.duration_s for c in programs)
    )
    if ratio < 0.05:
        return "thin interstitial inventory; breaks will repeat"
    return "ready to schedule"


def _chance(config: SchedulingConfig, position_s: float, probability: float) -> bool:
    """Deterministic pseudo-random decision.

    Seeded from the position so the same library always produces the same
    schedule -- reproducibility beats novelty for something an operator has to
    debug.
    """
    generator = random.Random(f"{config.seed}:{position_s:.2f}")
    return generator.random() < probability


def schedule_statistics(blocks: Sequence[ScheduleBlock]) -> dict[str, object]:
    total = sum(block.total_duration_s for block in blocks)
    pods = [pod for block in blocks for pod in block.pods]
    spots = [clip for pod in pods for clip in pod.clips]
    return {
        "blocks": len(blocks),
        "pods": len(pods),
        "spots": len(spots),
        "total_duration_s": round(total, 1),
        "total_hours": round(total / 3600.0, 2),
        "mean_pod_duration_s": round(
            sum(pod.duration_s for pod in pods) / len(pods), 1
        )
        if pods
        else 0.0,
        "unique_spots": len({clip.id for clip in spots}),
    }
