"""The search service: filters, keywords and vectors, fused.

A query runs three ways and the results are merged with Reciprocal Rank
Fusion. RRF is used rather than a weighted score blend because keyword scores
(BM25) and vector scores (cosine) live on incomparable scales -- normalizing
them against each other produces a number that looks principled and behaves
badly. RRF only uses each result's *rank*, which is exactly the property that
transfers between the two.

Structured filters (category, network, year, entity) are applied as a hard
gate, because "every Taco Bell commercial" means commercials, not
commercial-ish things.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from ..config import Settings
from ..constants import RRF_K, SEARCH_MAX_LIMIT
from ..db.base import Database
from ..db.models import Clip, ClipEntity
from ..db.repo import EntityRepository, TagRepository
from ..errors import BackendUnavailableError
from ..types import Category
from ..util.text import canonical_key
from . import vectors
from .fts import KeywordBackend, get_backend
from .query import ParsedQuery, parse

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    """One result, with why it matched."""

    clip_id: str
    score: float
    clip: Optional[Clip] = None
    matched_by: list[str] = field(default_factory=list)
    keyword_rank: Optional[int] = None
    vector_rank: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "clip_id": self.clip_id,
            "score": round(self.score, 4),
            "matched_by": self.matched_by,
        }
        if self.clip is not None:
            data.update(
                {
                    "title": self.clip.title,
                    "category": self.clip.category,
                    "subcategory": self.clip.subcategory,
                    "network": self.clip.network,
                    "series": self.clip.series,
                    "air_year": self.clip.air_year,
                    "duration_s": round(self.clip.duration_s, 2),
                    "summary": self.clip.summary,
                    "thumbnail": self.clip.thumbnail_path,
                    "path": self.clip.file_path,
                    "is_alternate": self.clip.is_alternate,
                    "confidence": round(self.clip.category_confidence, 3),
                }
            )
        return data


@dataclass
class SearchResponse:
    parsed: ParsedQuery
    hits: list[SearchHit]
    total_candidates: int
    semantic_available: bool
    explanation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.parsed.as_dict(),
            "explanation": self.explanation,
            "semantic_available": self.semantic_available,
            "total_candidates": self.total_candidates,
            "count": len(self.hits),
            "hits": [hit.as_dict() for hit in self.hits],
        }


class SearchService:
    """Answers questions about the library."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.keyword: KeywordBackend = get_backend(db)

    # -- public API ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: Optional[int] = None,
        include_alternates: bool = False,
        semantic: bool = True,
    ) -> SearchResponse:
        """Run a natural-language query."""
        resolved_limit = min(limit or self.settings.search.default_limit, SEARCH_MAX_LIMIT)
        parsed = parse(query, limit=resolved_limit)
        parsed.include_alternates = include_alternates

        if parsed.similar_to_clip_id:
            return self._similar(parsed)

        with self.db.session() as session:
            allowed = self._filter_ids(session, parsed)

        keyword_hits: list[tuple[str, float]] = []
        vector_hits: list[tuple[str, float]] = []
        semantic_available = False

        if parsed.text:
            with self.db.session() as session:
                keyword_hits = self.keyword.search(
                    session, parsed.text, limit=resolved_limit * 4
                )
            if semantic and self.settings.search.enabled:
                try:
                    query_vector = vectors.embed_query(self.settings, parsed.text)
                    vector_hits = vectors.search(
                        self.db, query_vector, limit=resolved_limit * 4
                    )
                    semantic_available = True
                except BackendUnavailableError as exc:
                    log.debug("semantic search unavailable: %s", exc)

        fused = self._fuse(keyword_hits, vector_hits, allowed, parsed)
        hits = self._hydrate(fused[:resolved_limit])

        explanation = self._explain(parsed, len(allowed) if allowed is not None else -1,
                                    bool(keyword_hits), semantic_available)
        return SearchResponse(
            parsed=parsed,
            hits=hits,
            total_candidates=len(allowed) if allowed is not None else len(fused),
            semantic_available=semantic_available,
            explanation=explanation,
        )

    def similar(self, clip_id: str, *, limit: int = 20) -> SearchResponse:
        """Clips like this one."""
        parsed = ParsedQuery(raw=f"similar to {clip_id}", similar_to_clip_id=clip_id, limit=limit)
        return self._similar(parsed)

    # -- internals ----------------------------------------------------------

    def _similar(self, parsed: ParsedQuery) -> SearchResponse:
        clip_id = parsed.similar_to_clip_id or ""
        pairs = vectors.similar_to_clip(
            self.db,
            clip_id,
            limit=parsed.limit,
            min_score=self.settings.search.similarity_min_score,
        )
        hits = self._hydrate(
            [SearchHit(clip_id=cid, score=score, matched_by=["vector"]) for cid, score in pairs]
        )
        explanation = (
            f"{len(hits)} clip(s) semantically similar to {clip_id[:8]}"
            if hits
            else f"no embedding stored for {clip_id[:8]} (run `bb reanalyze --missing`)"
        )
        return SearchResponse(
            parsed=parsed,
            hits=hits,
            total_candidates=len(pairs),
            semantic_available=True,
            explanation=explanation,
        )

    def _filter_ids(self, session: Session, parsed: ParsedQuery) -> Optional[set[str]]:
        """Apply structured filters, returning the allowed clip ids.

        ``None`` means "no structured filter was given", which is different
        from an empty set (a filter that matched nothing).
        """
        if not parsed.has_filters:
            return None

        conditions = []
        if parsed.category is not None:
            conditions.append(Clip.category == parsed.category.value)
        if parsed.network:
            conditions.append(Clip.network == parsed.network)
        if parsed.year_range:
            low, high = parsed.year_range
            conditions.append(and_(Clip.air_year >= low, Clip.air_year <= high))
        elif parsed.years:
            conditions.append(Clip.air_year.in_(parsed.years))
        if parsed.min_duration_s is not None:
            conditions.append(Clip.duration_s >= parsed.min_duration_s)
        if parsed.max_duration_s is not None:
            conditions.append(Clip.duration_s <= parsed.max_duration_s)
        if not parsed.include_alternates:
            conditions.append(Clip.is_alternate.is_(False))

        statement = select(Clip.id)
        if conditions:
            statement = statement.where(and_(*conditions))
        allowed = {str(row) for row in session.execute(statement).scalars()}

        for kind, name in parsed.entities:
            entity_ids = [
                entity.id
                for entity in EntityRepository.search(session, name, kind=kind, limit=10)
                if canonical_key(entity.name) == canonical_key(name)
                or canonical_key(name) in canonical_key(entity.name)
            ]
            if not entity_ids:
                # Named something we have never seen: fall back to text so the
                # query still has a chance rather than returning nothing.
                parsed.text = f"{parsed.text} {name}".strip()
                continue
            matching = {
                str(row)
                for row in session.execute(
                    select(ClipEntity.clip_id).where(ClipEntity.entity_id.in_(entity_ids))
                ).scalars()
            }
            allowed &= matching
        return allowed

    def _fuse(
        self,
        keyword_hits: Sequence[tuple[str, float]],
        vector_hits: Sequence[tuple[str, float]],
        allowed: Optional[set[str]],
        parsed: ParsedQuery,
    ) -> list[SearchHit]:
        """Reciprocal Rank Fusion of the two result lists."""
        k = self.settings.search.rrf_k or RRF_K
        scores: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        keyword_rank: dict[str, int] = {}
        vector_rank: dict[str, int] = {}

        for rank, (clip_id, _score) in enumerate(keyword_hits, start=1):
            scores[clip_id] = scores.get(clip_id, 0.0) + 1.0 / (k + rank)
            matched.setdefault(clip_id, []).append("keyword")
            keyword_rank[clip_id] = rank

        for rank, (clip_id, _score) in enumerate(vector_hits, start=1):
            scores[clip_id] = scores.get(clip_id, 0.0) + 1.0 / (k + rank)
            matched.setdefault(clip_id, []).append("vector")
            vector_rank[clip_id] = rank

        if not scores and allowed is not None:
            # A pure filter query: no text at all, so everything that passes
            # the filter is a hit, newest first.
            return [
                SearchHit(clip_id=clip_id, score=1.0, matched_by=["filter"])
                for clip_id in sorted(allowed)
            ]

        if allowed is not None:
            scores = {cid: value for cid, value in scores.items() if cid in allowed}

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            SearchHit(
                clip_id=clip_id,
                score=score,
                matched_by=matched.get(clip_id, []),
                keyword_rank=keyword_rank.get(clip_id),
                vector_rank=vector_rank.get(clip_id),
            )
            for clip_id, score in ordered
        ]

    def _hydrate(self, hits: Sequence[SearchHit]) -> list[SearchHit]:
        """Attach clip rows to hits in one query."""
        if not hits:
            return []
        ids = [hit.clip_id for hit in hits]
        with self.db.session() as session:
            rows = {
                clip.id: clip
                for clip in session.execute(select(Clip).where(Clip.id.in_(ids))).scalars()
            }
            for clip in rows.values():
                session.expunge(clip)
        for hit in hits:
            hit.clip = rows.get(hit.clip_id)
        return [hit for hit in hits if hit.clip is not None]

    def _explain(
        self, parsed: ParsedQuery, candidates: int, had_keywords: bool, semantic: bool
    ) -> str:
        parts = [f"interpreted as: {parsed.describe()}"]
        if candidates >= 0:
            parts.append(f"{candidates} clip(s) passed the filters")
        methods = [m for m, on in (("keyword", had_keywords), ("semantic", semantic)) if on]
        if methods:
            parts.append("ranked by " + " + ".join(methods))
        elif parsed.text:
            parts.append("no text matches found")
        return "; ".join(parts)

    # -- convenience --------------------------------------------------------

    def facets(self) -> dict[str, Any]:
        """Counts to drive the dashboard's filter chips."""
        from ..db.repo import ClipRepository

        with self.db.session() as session:
            by_category = ClipRepository.counts_by_category(session)
            networks = [
                (str(name), int(count))
                for name, count in session.execute(
                    select(Clip.network, func.count())
                    .where(Clip.network.is_not(None))
                    .group_by(Clip.network)
                ).all()
            ]
            years = [
                (int(year), int(count))
                for year, count in session.execute(
                    select(Clip.air_year, func.count())
                    .where(Clip.air_year.is_not(None))
                    .group_by(Clip.air_year)
                ).all()
            ]
            tags = TagRepository.popular(session, limit=40)
        return {
            "categories": by_category,
            "networks": sorted(networks, key=lambda kv: kv[1], reverse=True),
            "years": sorted(years),
            "tags": tags,
            "valid_categories": [c.value for c in Category],
        }
