"""Categorization: turning analyzer votes into a decision."""

from __future__ import annotations

from .classifier import Categorizer
from .rules import (
    FLAG_CATEGORIES,
    MIN_DECISION_CONFIDENCE,
    Decision,
    decide,
    merge_votes,
    subcategory_for,
    tally,
)

__all__ = [
    "FLAG_CATEGORIES",
    "MIN_DECISION_CONFIDENCE",
    "Categorizer",
    "Decision",
    "decide",
    "merge_votes",
    "subcategory_for",
    "tally",
]
