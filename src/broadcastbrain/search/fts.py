"""Keyword search index.

SQLite FTS5 when available, a LIKE-based fallback when not, behind one
protocol. The fallback exists because a SQLite built without FTS5 is not that
rare on stripped-down systems, and search degrading in speed is acceptable
where search failing outright is not.

The PostgreSQL implementation would be a third class here using ``tsvector``;
the protocol is the seam that keeps that change local.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from ..db.base import Database
from ..db.models import Clip, ClipEntity, ClipTag, ClipText, Entity
from ..db.schema import FTS_TABLE, fts_available
from .query import to_fts_expression

log = logging.getLogger(__name__)


class KeywordBackend(Protocol):
    """What the search service needs from a keyword index."""

    def index_clip(self, session: Session, clip_id: str) -> None: ...

    def remove_clip(self, session: Session, clip_id: str) -> None: ...

    def search(self, session: Session, query: str, *, limit: int) -> list[tuple[str, float]]: ...

    def rebuild(self, db: Database) -> int: ...

    @property
    def name(self) -> str: ...


def build_document(session: Session, clip_id: str) -> Optional[dict[str, str]]:
    """Gather every searchable string for one clip into indexable columns."""
    clip = session.get(Clip, clip_id)
    if clip is None:
        return None

    texts = {
        row.kind: row.content
        for row in session.query(ClipText).filter(ClipText.clip_id == clip_id).all()
    }
    tags = [
        row.tag for row in session.query(ClipTag).filter(ClipTag.clip_id == clip_id).all()
    ]
    entity_names = [
        name
        for (name,) in session.query(Entity.name)
        .join(ClipEntity, ClipEntity.entity_id == Entity.id)
        .filter(ClipEntity.clip_id == clip_id)
        .all()
    ]

    return {
        "clip_id": clip.id,
        "title": clip.title or "",
        "series": clip.series or "",
        "network": clip.network or "",
        "summary": clip.summary or "",
        "transcript": texts.get("transcript", ""),
        "ocr": texts.get("ocr", ""),
        "tags": " ".join(tags),
        "entities": " ".join(entity_names),
    }


class Fts5Backend:
    """FTS5-backed keyword index."""

    name = "fts5"

    def index_clip(self, session: Session, clip_id: str) -> None:
        document = build_document(session, clip_id)
        if document is None:
            return
        # FTS5 has no upsert; delete-then-insert is the documented pattern.
        self.remove_clip(session, clip_id)
        session.execute(
            sql_text(
                f"INSERT INTO {FTS_TABLE} "
                "(clip_id, title, series, network, summary, transcript, ocr, tags, entities) "
                "VALUES (:clip_id, :title, :series, :network, :summary, :transcript, :ocr, :tags, :entities)"
            ),
            document,
        )

    def remove_clip(self, session: Session, clip_id: str) -> None:
        session.execute(
            sql_text(f"DELETE FROM {FTS_TABLE} WHERE clip_id = :clip_id"), {"clip_id": clip_id}
        )

    def search(self, session: Session, query: str, *, limit: int) -> list[tuple[str, float]]:
        expression = to_fts_expression(query)
        if not expression:
            return []
        try:
            rows = session.execute(
                sql_text(
                    f"SELECT clip_id, bm25({FTS_TABLE}) AS score FROM {FTS_TABLE} "
                    f"WHERE {FTS_TABLE} MATCH :expression ORDER BY score LIMIT :limit"
                ),
                {"expression": expression, "limit": limit},
            ).all()
        except Exception as exc:
            log.debug("FTS query failed (%s); falling back to LIKE", exc)
            return LikeBackend().search(session, query, limit=limit)
        # bm25 returns lower-is-better; invert into a 0-1 relevance score.
        return [(str(clip_id), 1.0 / (1.0 + abs(float(score)))) for clip_id, score in rows]

    def rebuild(self, db: Database) -> int:
        """Reindex every clip from scratch."""
        count = 0
        with db.write_session() as session:
            session.execute(sql_text(f"DELETE FROM {FTS_TABLE}"))
            clip_ids = [str(row) for row in session.query(Clip.id).all()]
        for clip_id in clip_ids:
            with db.write_session() as session:
                self.index_clip(session, clip_id)
            count += 1
        log.info("keyword index rebuilt: %d clips", count)
        return count


class LikeBackend:
    """Portable fallback: substring matching over the clips table.

    Slower and cruder, but it works on any SQLite build and on PostgreSQL
    without extensions, and it keeps ``bb search`` honest on a machine where
    FTS5 is missing.
    """

    name = "like"

    def index_clip(self, session: Session, clip_id: str) -> None:
        return  # nothing to maintain

    def remove_clip(self, session: Session, clip_id: str) -> None:
        return

    def search(self, session: Session, query: str, *, limit: int) -> list[tuple[str, float]]:
        terms = [t for t in query.split() if t]
        if not terms:
            return []
        scores: dict[str, float] = {}

        for term in terms:
            pattern = f"%{term.lower()}%"
            for clip_id, weight in self._matches(session, pattern):
                scores[clip_id] = scores.get(clip_id, 0.0) + weight

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        highest = ordered[0][1] if ordered else 1.0
        return [(clip_id, score / highest) for clip_id, score in ordered]

    def _matches(self, session: Session, pattern: str) -> list[tuple[str, float]]:
        results: list[tuple[str, float]] = []
        # Weighted by field: a title hit means more than a transcript hit.
        for column, weight in (
            (Clip.title, 3.0),
            (Clip.series, 2.5),
            (Clip.network, 2.0),
            (Clip.summary, 1.5),
        ):
            rows = session.query(Clip.id).filter(column.ilike(pattern)).limit(500).all()
            results.extend((str(row[0]), weight) for row in rows)

        rows = (
            session.query(ClipText.clip_id)
            .filter(ClipText.content.ilike(pattern))
            .limit(500)
            .all()
        )
        results.extend((str(row[0]), 1.0) for row in rows)

        rows = session.query(ClipTag.clip_id).filter(ClipTag.tag.ilike(pattern)).limit(500).all()
        results.extend((str(row[0]), 2.0) for row in rows)
        return results

    def rebuild(self, db: Database) -> int:
        return 0


def get_backend(db: Database) -> KeywordBackend:
    """Pick the best keyword backend this database supports."""
    if db.is_sqlite and fts_available(db):
        return Fts5Backend()
    return LikeBackend()


def index_clips(db: Database, clip_ids: Sequence[str], backend: Optional[KeywordBackend] = None) -> int:
    """Index a batch of clips. Called after materialization."""
    resolved = backend or get_backend(db)
    indexed = 0
    with db.write_session() as session:
        for clip_id in clip_ids:
            resolved.index_clip(session, clip_id)
            indexed += 1
    return indexed
