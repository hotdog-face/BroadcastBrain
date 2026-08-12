"""Vote aggregation: deciding what a clip *is*.

Analyzers do not classify. They vote, with a weight and a stated reason, and
this module tallies the votes. Three properties fall out of that design:

* **Explainability.** Every clip stores the votes that produced its label, so
  "why is this a promo?" is answerable from the database alone.
* **Additivity.** A new analyzer improves classification by voting; it does not
  need to know about any other analyzer or about the decision rules.
* **Testability.** Everything here is pure functions over a list of tuples, so
  the classification logic is tested without ffmpeg, models or a database.

Confidence is the winner's *share* of the total vote weight, tempered by how
much evidence there was overall. A clip with one weak vote is not 100%
confident just because nothing disagreed with it.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..types import Category, Evidence

#: Below this confidence a clip is labelled UNKNOWN rather than guessed at.
#: Unknown clips are kept, exported and searchable -- they are simply honest.
MIN_DECISION_CONFIDENCE = 0.30

#: Scale of the "is there enough evidence at all?" curve. A winning weight of
#: roughly this much counts as substantial support; below it, confidence is
#: pulled down sharply no matter how uncontested the winner was.
EVIDENCE_SATURATION = 1.0

#: Categories that imply each boolean flag on the clip row.
FLAG_CATEGORIES: dict[str, frozenset[Category]] = {
    "is_commercial": frozenset({Category.COMMERCIAL, Category.INFOMERCIAL}),
    "is_promo": frozenset({Category.PROMO, Category.PPV_PREVIEW}),
    "is_episode": frozenset({Category.TV_EPISODE}),
    "is_movie": frozenset({Category.MOVIE}),
    "is_music_video": frozenset({Category.MUSIC_VIDEO}),
}

#: When two categories are close, the more specific one wins. Ties between
#: these pairs are resolved toward the first entry.
SPECIFICITY_ORDER: tuple[Category, ...] = (
    Category.STATION_ID,
    Category.NETWORK_BUMPER,
    Category.PPV_PREVIEW,
    Category.TV_GUIDE,
    Category.WEATHER,
    Category.NEWS_BREAK,
    Category.INFOMERCIAL,
    Category.MUSIC_VIDEO,
    Category.PROMO,
    Category.COMMERCIAL,
    Category.MOVIE,
    Category.TV_EPISODE,
    Category.UNKNOWN,
)
_SPECIFICITY = {category: index for index, category in enumerate(SPECIFICITY_ORDER)}


@dataclass
class Decision:
    """The classification outcome for one clip."""

    category: Category
    confidence: float
    evidence: list[Evidence] = field(default_factory=list)
    runner_up: Optional[Category] = None
    runner_up_confidence: float = 0.0
    totals: dict[str, float] = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        return self.confidence >= MIN_DECISION_CONFIDENCE

    def flags(self) -> dict[str, bool]:
        """Boolean columns implied by the chosen category."""
        return {
            flag: self.category in categories for flag, categories in FLAG_CATEGORIES.items()
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "confidence": round(self.confidence, 4),
            "runner_up": self.runner_up.value if self.runner_up else None,
            "runner_up_confidence": round(self.runner_up_confidence, 4),
            "totals": {k: round(v, 4) for k, v in self.totals.items()},
            "evidence": [e.as_dict() for e in self.evidence],
        }


def tally(votes: Iterable[tuple[Category, float, str]]) -> dict[Category, float]:
    """Sum vote weights per category.

    Weights are summed rather than maximized: three independent weak signals
    pointing at "commercial" should beat one moderate signal pointing
    elsewhere, which is exactly how this material behaves in practice.
    """
    totals: dict[Category, float] = defaultdict(float)
    for category, weight, _reason in votes:
        if weight > 0:
            totals[category] += float(weight)
    return dict(totals)


def decide(
    votes: Sequence[tuple[Category, float, str]],
    *,
    locked_category: Optional[Category] = None,
) -> Decision:
    """Choose a category from weighted votes.

    Args:
        locked_category: An operator override. Returned verbatim at full
            confidence -- automation never argues with a human decision.
    """
    evidence = [
        Evidence(category=category, weight=weight, reason=reason, source=_source_of(reason))
        for category, weight, reason in votes
        if weight > 0
    ]

    if locked_category is not None:
        return Decision(
            category=locked_category,
            confidence=1.0,
            evidence=evidence,
            totals={locked_category.value: 1.0},
        )

    totals = tally(votes)
    if not totals:
        return Decision(
            category=Category.UNKNOWN,
            confidence=0.0,
            evidence=evidence,
            totals={},
        )

    ordered = sorted(
        totals.items(),
        key=lambda item: (item[1], -_SPECIFICITY.get(item[0], 99)),
        reverse=True,
    )
    winner, winner_weight = ordered[0]
    runner_up, runner_up_weight = (ordered[1] if len(ordered) > 1 else (None, 0.0))

    total_weight = sum(totals.values())
    share = winner_weight / total_weight if total_weight else 0.0

    # Confidence is the winner's *share* of the vote, gated by how much
    # evidence there was in total. These two must multiply, not add: a single
    # feeble vote has a 100% share simply because nothing contradicted it, and
    # adding the terms would turn that into a confident answer. Multiplying
    # means being unopposed is worthless without substance behind it.
    saturation = 1.0 - math.exp(-winner_weight / EVIDENCE_SATURATION)
    confidence = round(min(0.99, share * saturation), 4)

    category = winner if confidence >= MIN_DECISION_CONFIDENCE else Category.UNKNOWN

    return Decision(
        category=category,
        confidence=confidence,
        evidence=evidence,
        runner_up=runner_up,
        runner_up_confidence=round(
            (runner_up_weight / total_weight) if total_weight else 0.0, 4
        ),
        totals={k.value: v for k, v in totals.items()},
    )


def subcategory_for(decision: Decision, duration_s: float, tags: Sequence[str] = ()) -> Optional[str]:
    """A finer label used for library subfoldering and scheduling.

    Ad slot length is the one that matters most for Fieldstation: a scheduler
    filling a two-minute pod needs to know it has 30s spots, not just "some
    commercials".
    """
    tag_set = {t.lower() for t in tags}

    if decision.category is Category.COMMERCIAL:
        for slot in (10, 15, 20, 30, 45, 60):
            if abs(duration_s - slot) <= 1.5:
                return f"{slot}s"
        return "other-length"

    if decision.category is Category.TV_EPISODE:
        if duration_s <= 1500:
            return "half-hour"
        if duration_s <= 2900:
            return "hour"
        return "extended"

    if decision.category is Category.MOVIE:
        return "letterboxed" if "letterboxed" in tag_set else "full-frame"

    if decision.category is Category.PROMO:
        return "tune-in" if duration_s <= 20 else "extended"

    if decision.category in (Category.NETWORK_BUMPER, Category.STATION_ID):
        return "animated" if "animation" in tag_set else "live"

    return None


def merge_votes(*vote_groups: Sequence[tuple[Category, float, str]]) -> list[tuple[Category, float, str]]:
    """Concatenate vote lists from several stages."""
    merged: list[tuple[Category, float, str]] = []
    for group in vote_groups:
        merged.extend(group)
    return merged


def _source_of(reason: str) -> str:
    """Recover the analyzer name from a ``"analyzer: reason"`` string."""
    head, separator, _tail = reason.partition(":")
    return head.strip() if separator else "rules"
