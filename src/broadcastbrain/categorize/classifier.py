"""Applying a classification decision to a clip.

Reads the stored analyzer results, replays their votes through
:mod:`broadcastbrain.categorize.rules`, adds the structural votes that only
this stage can know about (the clip's position in its source, its neighbours),
and writes the verdict.

The positional votes are worth calling out. A 30-second clip sitting between
two other 30-second clips, all inside a two-minute run, is a commercial pod --
and knowing that is often more reliable than anything visible inside any one of
those clips. Broadcast structure is itself evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from sqlalchemy.orm import Session

from ..analyze.runner import VOTES_KEY
from ..config import Settings
from ..constants import COMMERCIAL_SLOT_SECONDS, COMMERCIAL_SLOT_TOLERANCE_S
from ..db.base import Database
from ..db.models import Clip
from ..db.repo import AnalysisRepository, ClipRepository, TagRepository
from ..types import AnalysisStatus, Category, Stage
from .rules import Decision, decide, subcategory_for

log = logging.getLogger(__name__)

#: A run of this many similar-length short clips is almost certainly a pod.
POD_MIN_RUN = 2
#: Neighbour votes are deliberately modest; they support, never overrule.
NEIGHBOUR_VOTE_WEIGHT = 0.35


class Categorizer:
    """Decides each clip's category and writes it, with its evidence."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db

    def categorize(
        self, clip_id: str, extra_votes: Optional[Sequence[tuple[Category, float, str]]] = None
    ) -> Optional[Decision]:
        """Classify one clip. Returns ``None`` if the clip has vanished."""
        with self.db.session() as session:
            clip = ClipRepository.get(session, clip_id)
            if clip is None:
                return None
            stored_votes = _votes_from_analyses(session, clip_id)
            tags = TagRepository.for_clip(session, clip_id)
            siblings = ClipRepository.by_source(session, clip.source_id)
            # Detach everything at once: `clip` is itself one of its siblings,
            # so expunging them individually would double-expunge it.
            session.expunge_all()

        votes = list(stored_votes)
        votes.extend(extra_votes or ())
        votes.extend(self._positional_votes(clip, siblings))

        locked = Category(clip.category) if clip.locked else None
        decision = decide(votes, locked_category=locked)
        subcategory = subcategory_for(decision, clip.duration_s, tags)

        with self.db.write_session() as session:
            ClipRepository.update_fields(
                session,
                clip_id,
                category=decision.category.value,
                category_confidence=decision.confidence,
                category_evidence=[e.as_dict() for e in decision.evidence],
                subcategory=subcategory,
                **decision.flags(),
            )
            ClipRepository.set_stage(session, clip_id, Stage.CATEGORIZED)

        log.debug(
            "clip %s -> %s (%.2f) via %d votes",
            clip_id[:8], decision.category.value, decision.confidence, len(decision.evidence),
        )
        return decision

    # -- structural evidence ------------------------------------------------

    def _positional_votes(
        self, clip: Clip, siblings: Sequence[Clip]
    ) -> list[tuple[Category, float, str]]:
        """Votes derived from where the clip sits inside its recording."""
        votes: list[tuple[Category, float, str]] = []
        if len(siblings) < 2:
            return votes

        ordered = sorted(siblings, key=lambda c: c.seq)
        index = next((i for i, c in enumerate(ordered) if c.id == clip.id), None)
        if index is None:
            return votes

        # 1. Commercial pod: a run of ad-slot-length clips around this one.
        run_length = _pod_run_length(ordered, index)
        if run_length >= POD_MIN_RUN and _is_slot_length(clip.duration_s):
            votes.append(
                (
                    Category.COMMERCIAL,
                    min(0.7, NEIGHBOUR_VOTE_WEIGHT + 0.15 * run_length),
                    f"structure: inside a run of {run_length + 1} ad-length clips",
                )
            )

        # 2. First clip of a long recording: usually mid-program, not an ad.
        if index == 0 and clip.duration_s > 300:
            votes.append(
                (Category.TV_EPISODE, 0.2, "structure: opening segment of the recording")
            )

        # 3. A very short clip wedged between two long ones is an interstitial.
        if 0 < index < len(ordered) - 1 and clip.duration_s <= 12:
            before, after = ordered[index - 1], ordered[index + 1]
            if before.duration_s > 120 and after.duration_s > 120:
                votes.append(
                    (
                        Category.NETWORK_BUMPER,
                        0.4,
                        "structure: brief clip between two long segments",
                    )
                )
        return votes


def _pod_run_length(ordered: Sequence[Clip], index: int) -> int:
    """How many consecutive ad-slot-length neighbours surround ``index``."""
    run = 0
    for step in (-1, 1):
        cursor = index + step
        while 0 <= cursor < len(ordered) and _is_slot_length(ordered[cursor].duration_s):
            run += 1
            cursor += step
    return run


def _is_slot_length(duration_s: float) -> bool:
    return any(
        abs(duration_s - slot) <= COMMERCIAL_SLOT_TOLERANCE_S for slot in COMMERCIAL_SLOT_SECONDS
    )


def _votes_from_analyses(session: Session, clip_id: str) -> list[tuple[Category, float, str]]:
    """Recover category votes from stored analyzer payloads.

    Each analyzer's votes were persisted inside its payload when it ran, so
    re-categorizing the entire library is a database sweep -- no model, no
    ffmpeg, no re-decoding. ``bb recategorize`` is exactly this path applied to
    every clip, and it takes seconds rather than a night.
    """
    votes: list[tuple[Category, float, str]] = []
    records = AnalysisRepository.for_clip(session, clip_id)

    for name, record in records.items():
        if record.status != AnalysisStatus.OK.value or not isinstance(record.payload, dict):
            continue
        for category, weight, reason in _replay(record.payload):
            votes.append((category, weight, f"{name}: {reason}"))
    return votes


def _replay(payload: dict[str, Any]) -> list[tuple[Category, float, str]]:
    """Decode the votes an analyzer stored under :data:`VOTES_KEY`."""
    stored = payload.get(VOTES_KEY)
    if not isinstance(stored, list):
        return []
    votes: list[tuple[Category, float, str]] = []
    for entry in stored:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            continue
        label, weight, reason = entry
        try:
            votes.append((Category(str(label)), float(weight), str(reason)))
        except (ValueError, TypeError):
            continue
    return votes
