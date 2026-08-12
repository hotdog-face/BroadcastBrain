"""Duplicate detection and resolution.

Three independent identity signals, applied cheapest-first:

1. **SHA256** -- byte-identical files. Free, exact, no false positives.
2. **Perceptual hash sequence** -- the same footage re-encoded, rescaled or
   captured from a different source. This is where the real duplicates in an
   archive live.
3. **Chromaprint** -- the same audio under different video treatment; catches
   a spot that was letterboxed in one copy and not the other.

Candidate selection is blocked by duration, which turns an O(n^2) sweep of the
whole library into a handful of comparisons per clip. Without that, a hundred
thousand clips would be five billion comparisons.

**Nothing is ever deleted.** The best copy becomes canonical; the rest are
marked as alternates and filed under ``Duplicates/<canonical-id>/`` rather than
in the library, so browsing ``Commercials/`` shows one copy of a spot instead of
four. Alternates keep their database rows and stay fully searchable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from ..config import Settings
from ..db.base import Database
from ..db.models import Clip
from ..db.repo import ClipRepository, DuplicateRepository, EventRepository
from ..errors import StorageError
from ..types import ClipStatus, DuplicateMethod
from ..util.atomic import atomic_move, safe_filename
from ..util.hashing import decode_hash_sequence, fingerprint_similarity
from .hashes import compare_sequences, head_distance

log = logging.getLogger(__name__)

#: If the first frames differ by more than this many bits, the clips are not
#: the same content and the full comparison is skipped.
HEAD_PREFILTER_BITS = 26


@dataclass
class DuplicateMatch:
    """One clip judged to be the same content as another."""

    clip_id: str
    similarity: float
    method: DuplicateMethod
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "similarity": round(self.similarity, 4),
            "method": self.method.value,
            "detail": self.detail,
        }


@dataclass
class DedupeOutcome:
    clip_id: str
    matches: list[DuplicateMatch] = field(default_factory=list)
    canonical_id: Optional[str] = None
    is_alternate: bool = False
    candidates_examined: int = 0

    @property
    def found_duplicates(self) -> bool:
        return bool(self.matches)

    def as_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "canonical_id": self.canonical_id,
            "is_alternate": self.is_alternate,
            "candidates_examined": self.candidates_examined,
            "matches": [m.as_dict() for m in self.matches],
        }


class Deduplicator:
    """Finds duplicates of a clip and decides which copy wins."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.cfg = settings.dedupe

    def process(self, clip_id: str) -> DedupeOutcome:
        """Compare one clip against the library and resolve any duplicates."""
        outcome = DedupeOutcome(clip_id=clip_id)
        if not self.cfg.enabled:
            return outcome

        with self.db.session() as session:
            clip = ClipRepository.get(session, clip_id)
            if clip is None:
                return outcome
            candidates = ClipRepository.dedupe_candidates(
                session, clip, tolerance_s=self.cfg.duration_tolerance_s
            )
            # Detach so comparison happens without holding the session.
            session.expunge(clip)
            for candidate in candidates:
                session.expunge(candidate)

        outcome.candidates_examined = len(candidates)
        outcome.matches = self._find_matches(clip, candidates)
        if not outcome.matches:
            return outcome

        self._resolve(clip, outcome)
        return outcome

    # -- matching -----------------------------------------------------------

    def _find_matches(self, clip: Clip, candidates: Sequence[Clip]) -> list[DuplicateMatch]:
        matches: list[DuplicateMatch] = []
        clip_hashes = decode_hash_sequence(clip.phash)

        for candidate in candidates:
            match = self._compare(clip, clip_hashes, candidate)
            if match is not None:
                matches.append(match)

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    def _compare(
        self, clip: Clip, clip_hashes: list[int], candidate: Clip
    ) -> Optional[DuplicateMatch]:
        # 1. Exact bytes.
        if clip.sha256 and candidate.sha256 and clip.sha256 == candidate.sha256:
            return DuplicateMatch(
                clip_id=candidate.id,
                similarity=1.0,
                method=DuplicateMethod.EXACT_SHA256,
                detail="identical file content",
            )

        # 2. Perceptual, with a cheap pre-filter on the opening frame.
        candidate_hashes = decode_hash_sequence(candidate.phash)
        if clip_hashes and candidate_hashes:
            head = head_distance(clip.phash_head, candidate.phash_head)
            if head is None or head <= HEAD_PREFILTER_BITS:
                distance = compare_sequences(clip_hashes, candidate_hashes)
                if distance <= self.cfg.max_normalized_distance:
                    similarity = 1.0 - distance
                    # Corroborate with audio when both sides have it: it makes
                    # the difference between "same ad campaign" and "same spot".
                    audio_similarity = fingerprint_similarity(
                        clip.chromaprint, candidate.chromaprint
                    )
                    method = DuplicateMethod.PERCEPTUAL
                    if audio_similarity >= self.cfg.chromaprint_min_similarity:
                        similarity = min(1.0, (similarity + audio_similarity) / 2.0 + 0.05)
                        method = DuplicateMethod.COMPOSITE
                    return DuplicateMatch(
                        clip_id=candidate.id,
                        similarity=similarity,
                        method=method,
                        detail=f"phash distance {distance:.3f}, audio {audio_similarity:.2f}",
                    )

        # 3. Audio alone, for clips whose video treatment differs completely.
        if clip.chromaprint and candidate.chromaprint:
            audio_similarity = fingerprint_similarity(clip.chromaprint, candidate.chromaprint)
            if audio_similarity >= self.cfg.chromaprint_min_similarity:
                return DuplicateMatch(
                    clip_id=candidate.id,
                    similarity=audio_similarity,
                    method=DuplicateMethod.AUDIO_FINGERPRINT,
                    detail=f"audio fingerprint {audio_similarity:.2f}",
                )
        return None

    # -- resolution ---------------------------------------------------------

    def _resolve(self, clip: Clip, outcome: DedupeOutcome) -> None:
        """Pick the canonical copy and mark the others as alternates."""
        member_ids = [m.clip_id for m in outcome.matches]
        with self.db.write_session() as session:
            group_members = [ClipRepository.get(session, cid) for cid in member_ids]
            members = [m for m in group_members if m is not None]
            fresh_clip = ClipRepository.get(session, clip.id)
            if fresh_clip is None:  # pragma: no cover - deleted mid-flight
                return
            everyone = [fresh_clip, *members]

            canonical = max(everyone, key=_ranking_key)
            outcome.canonical_id = canonical.id
            outcome.is_alternate = canonical.id != fresh_clip.id

            similarity_by_id = {m.clip_id: m.similarity for m in outcome.matches}
            similarity_by_id[fresh_clip.id] = 1.0

            method = outcome.matches[0].method.value
            DuplicateRepository.record(
                session,
                canonical_clip_id=canonical.id,
                members=[(c.id, similarity_by_id.get(c.id, 1.0)) for c in everyone if c.id != canonical.id],
                method=method,
            )

            for candidate in everyone:
                if candidate.id == canonical.id:
                    if candidate.is_alternate:
                        # A previously demoted clip has been promoted because a
                        # better comparison arrived; restore it.
                        candidate.is_alternate = False
                        candidate.duplicate_of = None
                        candidate.status = ClipStatus.READY.value
                    continue
                candidate.is_alternate = True
                candidate.duplicate_of = canonical.id
                candidate.duplicate_similarity = similarity_by_id.get(candidate.id)
                candidate.status = ClipStatus.DUPLICATE.value

            EventRepository.log(
                session,
                "duplicate",
                f"{len(everyone)} copies resolved; kept {canonical.id[:8]} "
                f"(quality {canonical.quality_score:.3f})",
                clip_id=canonical.id,
                detail=outcome.as_dict(),
            )

        if self.cfg.move_alternates:
            self._relocate_alternates(outcome)

    def _relocate_alternates(self, outcome: DedupeOutcome) -> None:
        """Move demoted copies out of the library and into ``Duplicates/``.

        This only has work to do when a clip was already materialized and a
        *better* copy arrived afterwards. Clips demoted before materialization
        are filed correctly in the first place by the materializer.

        The alternate is moved, not copied: one clip, one home. It stays in the
        database, stays searchable, and is simply not what the library serves.
        """
        duplicates_dir = self.settings.paths.require("duplicates")
        canonical_id = outcome.canonical_id or "unresolved"

        for match in outcome.matches:
            with self.db.session() as session:
                clip = ClipRepository.get(session, match.clip_id)
                if clip is None or not clip.is_alternate or not clip.file_path:
                    continue
                current = Path(clip.file_path)

            if not current.exists():
                continue
            try:
                current.relative_to(duplicates_dir)
                continue  # already filed as a duplicate
            except ValueError:
                pass

            target = duplicates_dir / safe_filename(canonical_id) / current.name
            try:
                moved = atomic_move(current, target)
            except StorageError as exc:  # pragma: no cover - filesystem specific
                log.warning("could not relocate alternate %s: %s", match.clip_id[:8], exc)
                continue

            with self.db.write_session() as session:
                ClipRepository.update_fields(session, match.clip_id, file_path=str(moved))
            log.debug("alternate %s moved to %s", match.clip_id[:8], moved.parent)


def _ranking_key(clip: Clip) -> tuple[float, float, int, float]:
    """Order copies best-first.

    Quality score leads. Ties break on how *complete* the copy is (a longer cut
    of the same spot usually means the other was clipped), then on pixel count,
    then on how recently it was ingested.
    """
    return (
        float(clip.quality_score or 0.0),
        float(clip.duration_s or 0.0),
        int((clip.width or 0) * (clip.height or 0)),
        float(clip.created_at.timestamp()) if clip.created_at else 0.0,
    )


def duplicate_statistics(db: Database) -> dict[str, int]:
    with db.session() as session:
        return DuplicateRepository.statistics(session)
