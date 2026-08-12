"""Repositories: every database access pattern the pipeline needs.

Keeping queries here (rather than scattered through the stages) means the
indexes in :mod:`models` can be reasoned about against a finite, visible set of
access patterns -- which is what keeps the library fast at a hundred thousand
clips.

All methods take an explicit :class:`~sqlalchemy.orm.Session` so a caller can
compose several repository calls into one transaction.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from sqlalchemy import Select, delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..types import (
    AnalysisStatus,
    Category,
    ClipStatus,
    EntityKind,
    Segment,
    SourceKind,
    SourceStatus,
    Stage,
)
from ..util.text import canonical_key
from .models import (
    Analysis,
    Clip,
    ClipEntity,
    ClipTag,
    ClipText,
    DuplicateGroup,
    DuplicateMember,
    Embedding,
    Entity,
    Event,
    KeyValue,
    RunRecord,
    Source,
    utcnow,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class SourceRepository:
    """Source files: discovery, identity and lifecycle."""

    @staticmethod
    def get(session: Session, source_id: str) -> Optional[Source]:
        return session.get(Source, source_id)

    @staticmethod
    def by_sha256(session: Session, sha256: str) -> Optional[Source]:
        return session.execute(
            select(Source).where(Source.sha256 == sha256)
        ).scalar_one_or_none()

    @staticmethod
    def by_quick_signature(session: Session, signature: str) -> list[Source]:
        return list(
            session.execute(
                select(Source).where(Source.quick_signature == signature)
            ).scalars()
        )

    @staticmethod
    def by_path(session: Session, path: Path) -> Optional[Source]:
        return session.execute(
            select(Source).where(
                or_(Source.path == str(path), Source.original_path == str(path))
            )
        ).scalar_one_or_none()

    @staticmethod
    def create(
        session: Session,
        *,
        path: Path,
        size_bytes: int,
        mtime: Optional[datetime] = None,
        quick_signature: Optional[str] = None,
        sha256: Optional[str] = None,
        kind: SourceKind = SourceKind.UNKNOWN,
        origin: Optional[str] = None,
        origin_ref: Optional[str] = None,
        parent_source_id: Optional[str] = None,
    ) -> Source:
        source = Source(
            path=str(path),
            original_path=str(path),
            filename=path.name,
            size_bytes=size_bytes,
            mtime=mtime,
            quick_signature=quick_signature,
            sha256=sha256,
            kind=kind.value,
            origin=origin,
            origin_ref=origin_ref,
            parent_source_id=parent_source_id,
            status=SourceStatus.DISCOVERED.value,
            stage=Stage.DISCOVERED.value,
        )
        session.add(source)
        session.flush()
        return source

    @staticmethod
    def set_stage(session: Session, source_id: str, stage: Stage) -> None:
        session.execute(
            update(Source).where(Source.id == source_id).values(stage=stage.value)
        )

    @staticmethod
    def set_status(
        session: Session,
        source_id: str,
        status: SourceStatus,
        *,
        error: Optional[str] = None,
        reject_reason: Optional[str] = None,
    ) -> None:
        values: dict[str, Any] = {"status": status.value}
        if error is not None:
            values["error"] = error[:8000]
        if reject_reason is not None:
            values["reject_reason"] = reject_reason[:2000]
        if status is SourceStatus.COMPLETE:
            values["completed_at"] = utcnow()
        session.execute(update(Source).where(Source.id == source_id).values(**values))

    @staticmethod
    def list_by_status(
        session: Session, status: SourceStatus, *, limit: int = 100
    ) -> list[Source]:
        return list(
            session.execute(
                select(Source)
                .where(Source.status == status.value)
                .order_by(Source.created_at.asc())
                .limit(limit)
            ).scalars()
        )

    @staticmethod
    def unfinished(session: Session, *, limit: int = 1000) -> list[Source]:
        """Sources that were discovered or partly processed but never finished.

        This query is the whole of "resume where it left off" at the source
        level; the orchestrator re-enqueues whatever it returns.
        """
        return list(
            session.execute(
                select(Source)
                .where(
                    Source.status.in_(
                        [SourceStatus.DISCOVERED.value, SourceStatus.PROCESSING.value]
                    )
                )
                .order_by(Source.created_at.asc())
                .limit(limit)
            ).scalars()
        )

    @staticmethod
    def counts_by_status(session: Session) -> dict[str, int]:
        rows = session.execute(
            select(Source.status, func.count()).group_by(Source.status)
        ).all()
        return {str(status): int(count) for status, count in rows}

    @staticmethod
    def total_bytes(session: Session) -> int:
        return int(session.execute(select(func.coalesce(func.sum(Source.size_bytes), 0))).scalar_one())


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------


class ClipRepository:
    """Clips: creation from segments, updates, and the dedupe/export queries."""

    @staticmethod
    def get(session: Session, clip_id: str) -> Optional[Clip]:
        return session.get(Clip, clip_id)

    @staticmethod
    def by_source(session: Session, source_id: str) -> list[Clip]:
        return list(
            session.execute(
                select(Clip).where(Clip.source_id == source_id).order_by(Clip.seq)
            ).scalars()
        )

    @staticmethod
    def upsert_segments(
        session: Session, source_id: str, segments: Sequence[Segment]
    ) -> list[Clip]:
        """Create clip rows for a source's segments, idempotently.

        Keyed on ``(source_id, seq)``: re-running segmentation after a crash
        updates the existing rows in place instead of creating a second set of
        clips, so the analysis already done against those ids is preserved.
        """
        existing = {clip.seq: clip for clip in ClipRepository.by_source(session, source_id)}
        clips: list[Clip] = []
        for segment in segments:
            clip = existing.get(segment.index)
            if clip is None:
                clip = Clip(
                    source_id=source_id,
                    seq=segment.index,
                    start_s=segment.start_s,
                    end_s=segment.end_s,
                    duration_s=segment.duration_s,
                    status=ClipStatus.NEW.value,
                    stage=Stage.SEGMENTED.value,
                )
                session.add(clip)
            else:
                clip.start_s = segment.start_s
                clip.end_s = segment.end_s
                clip.duration_s = segment.duration_s
            clips.append(clip)
        session.flush()
        return clips

    @staticmethod
    def set_stage(session: Session, clip_id: str, stage: Stage) -> None:
        session.execute(update(Clip).where(Clip.id == clip_id).values(stage=stage.value))

    @staticmethod
    def set_status(
        session: Session, clip_id: str, status: ClipStatus, *, error: Optional[str] = None
    ) -> None:
        values: dict[str, Any] = {"status": status.value}
        if error is not None:
            values["error"] = error[:8000]
        session.execute(update(Clip).where(Clip.id == clip_id).values(**values))

    @staticmethod
    def update_fields(session: Session, clip_id: str, **fields: Any) -> None:
        if fields:
            session.execute(update(Clip).where(Clip.id == clip_id).values(**fields))

    @staticmethod
    def at_stage(
        session: Session, stage: Stage, *, limit: int = 500, exclude_alternates: bool = False
    ) -> list[Clip]:
        stmt = select(Clip).where(Clip.stage == stage.value)
        if exclude_alternates:
            stmt = stmt.where(Clip.is_alternate.is_(False))
        return list(session.execute(stmt.order_by(Clip.created_at.asc()).limit(limit)).scalars())

    @staticmethod
    def incomplete(session: Session, *, limit: int = 5000) -> list[Clip]:
        """Clips that have not reached the end of the pipeline.

        Used on start-up to rebuild the queue after an interruption.
        """
        return list(
            session.execute(
                select(Clip)
                .where(
                    Clip.status.notin_([ClipStatus.FAILED.value]),
                    Clip.stage != Stage.EXPORTED.value,
                )
                .order_by(Clip.created_at.asc())
                .limit(limit)
            ).scalars()
        )

    @staticmethod
    def dedupe_candidates(
        session: Session, clip: Clip, *, tolerance_s: float, limit: int = 400
    ) -> list[Clip]:
        """Clips that *could* be duplicates of ``clip``.

        Blocking by duration turns an O(n^2) comparison across the whole
        library into a small candidate set. A 30-second Taco Bell spot is only
        ever compared against other ~30-second clips.
        """
        low = clip.duration_s - tolerance_s
        high = clip.duration_s + tolerance_s
        return list(
            session.execute(
                select(Clip)
                .where(
                    Clip.id != clip.id,
                    Clip.duration_s >= low,
                    Clip.duration_s <= high,
                    Clip.is_alternate.is_(False),
                )
                .order_by(Clip.created_at.asc())
                .limit(limit)
            ).scalars()
        )

    @staticmethod
    def by_sha256(session: Session, sha256: str, *, exclude_clip_id: Optional[str] = None) -> list[Clip]:
        stmt = select(Clip).where(Clip.sha256 == sha256)
        if exclude_clip_id:
            stmt = stmt.where(Clip.id != exclude_clip_id)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def ready_for_export(
        session: Session, *, min_confidence: float = 0.0, limit: Optional[int] = None
    ) -> list[Clip]:
        stmt = (
            select(Clip)
            .where(
                Clip.is_alternate.is_(False),
                Clip.status.in_([ClipStatus.READY.value]),
                Clip.file_path.is_not(None),
                Clip.category_confidence >= min_confidence,
            )
            .order_by(Clip.category, Clip.series, Clip.seq)
        )
        if limit:
            stmt = stmt.limit(limit)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def iter_all(session: Session, *, batch_size: int = 500) -> Iterator[Clip]:
        """Memory-bounded iteration over every clip (indexing, migration)."""
        yield from session.execute(
            select(Clip).order_by(Clip.created_at.asc()).execution_options(yield_per=batch_size)
        ).scalars()

    @staticmethod
    def counts_by_category(session: Session) -> dict[str, int]:
        rows = session.execute(
            select(Clip.category, func.count())
            .where(Clip.is_alternate.is_(False))
            .group_by(Clip.category)
        ).all()
        return {str(category): int(count) for category, count in rows}

    @staticmethod
    def statistics(session: Session) -> dict[str, Any]:
        total = int(session.execute(select(func.count()).select_from(Clip)).scalar_one())
        alternates = int(
            session.execute(
                select(func.count()).select_from(Clip).where(Clip.is_alternate.is_(True))
            ).scalar_one()
        )
        duration = float(
            session.execute(select(func.coalesce(func.sum(Clip.duration_s), 0.0))).scalar_one()
        )
        bytes_total = int(
            session.execute(select(func.coalesce(func.sum(Clip.size_bytes), 0))).scalar_one() or 0
        )
        unknown = int(
            session.execute(
                select(func.count())
                .select_from(Clip)
                .where(Clip.category == Category.UNKNOWN.value)
            ).scalar_one()
        )
        return {
            "total_clips": total,
            "alternates": alternates,
            "unique_clips": total - alternates,
            "unknown_clips": unknown,
            "total_duration_s": duration,
            "total_bytes": bytes_total,
            "by_category": ClipRepository.counts_by_category(session),
        }


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


class AnalysisRepository:
    """Per-analyzer results -- the cache that stops all redundant AI work."""

    @staticmethod
    def for_clip(session: Session, clip_id: str) -> dict[str, Analysis]:
        rows = session.execute(select(Analysis).where(Analysis.clip_id == clip_id)).scalars()
        return {row.analyzer: row for row in rows}

    @staticmethod
    def upsert(
        session: Session,
        clip_id: str,
        *,
        analyzer: str,
        analyzer_version: str,
        status: AnalysisStatus,
        payload: Optional[dict[str, Any]] = None,
        confidence: Optional[float] = None,
        backend: Optional[str] = None,
        error: Optional[str] = None,
        duration_ms: int = 0,
    ) -> Analysis:
        row = session.execute(
            select(Analysis).where(Analysis.clip_id == clip_id, Analysis.analyzer == analyzer)
        ).scalar_one_or_none()
        if row is None:
            row = Analysis(clip_id=clip_id, analyzer=analyzer)
            session.add(row)
        row.analyzer_version = analyzer_version
        row.status = status.value
        row.payload = payload
        row.confidence = confidence
        row.backend = backend
        row.error = error[:8000] if error else None
        row.duration_ms = duration_ms
        row.created_at = utcnow()
        session.flush()
        return row

    @staticmethod
    def clips_missing_analyzer(
        session: Session, analyzer: str, *, version: Optional[str] = None, limit: int = 1000
    ) -> list[str]:
        """Clip ids that lack a successful, current-version run of ``analyzer``.

        This single query drives ``bb reanalyze --missing``: install a backend
        that was absent last night, and only the clips that need it get redone.
        """
        done = select(Analysis.clip_id).where(
            Analysis.analyzer == analyzer,
            Analysis.status == AnalysisStatus.OK.value,
        )
        if version is not None:
            done = done.where(Analysis.analyzer_version == version)
        return [
            str(row)
            for row in session.execute(
                select(Clip.id).where(Clip.id.notin_(done)).limit(limit)
            ).scalars()
        ]

    @staticmethod
    def counts_by_status(session: Session) -> dict[str, dict[str, int]]:
        rows = session.execute(
            select(Analysis.analyzer, Analysis.status, func.count()).group_by(
                Analysis.analyzer, Analysis.status
            )
        ).all()
        out: dict[str, dict[str, int]] = {}
        for analyzer, status, count in rows:
            out.setdefault(str(analyzer), {})[str(status)] = int(count)
        return out

    @staticmethod
    def clear(session: Session, clip_id: str, *, analyzers: Optional[Sequence[str]] = None) -> int:
        stmt = delete(Analysis).where(Analysis.clip_id == clip_id)
        if analyzers:
            stmt = stmt.where(Analysis.analyzer.in_(list(analyzers)))
        return _rowcount(session.execute(stmt))


# ---------------------------------------------------------------------------
# Text, tags, entities, embeddings
# ---------------------------------------------------------------------------


class TextRepository:
    @staticmethod
    def upsert(
        session: Session,
        clip_id: str,
        *,
        kind: str,
        content: str,
        language: Optional[str] = None,
        detail: Optional[dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ) -> ClipText:
        row = session.execute(
            select(ClipText).where(ClipText.clip_id == clip_id, ClipText.kind == kind)
        ).scalar_one_or_none()
        if row is None:
            row = ClipText(clip_id=clip_id, kind=kind)
            session.add(row)
        row.content = content
        row.language = language
        row.detail = detail
        row.confidence = confidence
        session.flush()
        return row

    @staticmethod
    def get(session: Session, clip_id: str, kind: str) -> Optional[ClipText]:
        return session.execute(
            select(ClipText).where(ClipText.clip_id == clip_id, ClipText.kind == kind)
        ).scalar_one_or_none()

    @staticmethod
    def all_for_clip(session: Session, clip_id: str) -> dict[str, str]:
        rows = session.execute(select(ClipText).where(ClipText.clip_id == clip_id)).scalars()
        return {row.kind: row.content for row in rows}


class TagRepository:
    @staticmethod
    def set_tags(
        session: Session,
        clip_id: str,
        tags: Iterable[tuple[str, float]],
        *,
        source: str = "auto",
        replace_source: bool = True,
    ) -> int:
        """Replace this source's tags for a clip. Manual tags are untouched."""
        if replace_source:
            session.execute(
                delete(ClipTag).where(ClipTag.clip_id == clip_id, ClipTag.source == source)
            )
            session.flush()
        existing = {
            row.tag
            for row in session.execute(
                select(ClipTag).where(ClipTag.clip_id == clip_id)
            ).scalars()
        }
        added = 0
        for tag, confidence in tags:
            normalized = tag.strip().lower()[:128]
            if not normalized or normalized in existing:
                continue
            session.add(
                ClipTag(clip_id=clip_id, tag=normalized, confidence=confidence, source=source)
            )
            existing.add(normalized)
            added += 1
        session.flush()
        return added

    @staticmethod
    def for_clip(session: Session, clip_id: str) -> list[str]:
        return [
            str(tag)
            for tag in session.execute(
                select(ClipTag.tag).where(ClipTag.clip_id == clip_id).order_by(ClipTag.tag)
            ).scalars()
        ]

    @staticmethod
    def popular(session: Session, *, limit: int = 100) -> list[tuple[str, int]]:
        rows = session.execute(
            select(ClipTag.tag, func.count())
            .group_by(ClipTag.tag)
            .order_by(func.count().desc())
            .limit(limit)
        ).all()
        return [(str(tag), int(count)) for tag, count in rows]


class EntityRepository:
    """People, brands, products, shows, networks, songs."""

    @staticmethod
    def get_or_create(session: Session, kind: EntityKind, name: str) -> Entity:
        key = canonical_key(name)
        if not key:
            raise ValueError(f"entity name normalizes to nothing: {name!r}")
        entity = session.execute(
            select(Entity).where(Entity.kind == kind.value, Entity.canonical_key == key)
        ).scalar_one_or_none()
        if entity is not None:
            return entity
        entity = Entity(kind=kind.value, name=name.strip()[:256], canonical_key=key)
        session.add(entity)
        try:
            session.flush()
        except IntegrityError:  # pragma: no cover - concurrent create
            session.rollback()
            entity = session.execute(
                select(Entity).where(Entity.kind == kind.value, Entity.canonical_key == key)
            ).scalar_one()
        return entity

    @staticmethod
    def link(
        session: Session,
        clip_id: str,
        kind: EntityKind,
        name: str,
        *,
        confidence: float,
        analyzer: str,
        at_s: Optional[float] = None,
    ) -> Optional[ClipEntity]:
        try:
            entity = EntityRepository.get_or_create(session, kind, name)
        except ValueError:
            return None
        existing = session.execute(
            select(ClipEntity).where(
                ClipEntity.clip_id == clip_id,
                ClipEntity.entity_id == entity.id,
                ClipEntity.source_analyzer == analyzer,
            )
        ).scalar_one_or_none()
        if existing is not None:
            if confidence > existing.confidence:
                existing.confidence = confidence
                existing.at_s = at_s
            return existing
        link = ClipEntity(
            clip_id=clip_id,
            entity_id=entity.id,
            confidence=confidence,
            source_analyzer=analyzer,
            at_s=at_s,
        )
        session.add(link)
        entity.clip_count = (entity.clip_count or 0) + 1
        session.flush()
        return link

    @staticmethod
    def for_clip(session: Session, clip_id: str) -> list[tuple[Entity, float]]:
        rows = session.execute(
            select(Entity, ClipEntity.confidence)
            .join(ClipEntity, ClipEntity.entity_id == Entity.id)
            .where(ClipEntity.clip_id == clip_id)
            .order_by(ClipEntity.confidence.desc())
        ).all()
        return [(entity, float(confidence)) for entity, confidence in rows]

    @staticmethod
    def search(
        session: Session, name: str, *, kind: Optional[EntityKind] = None, limit: int = 25
    ) -> list[Entity]:
        key = canonical_key(name)
        stmt = select(Entity).where(Entity.canonical_key.like(f"%{key}%"))
        if kind is not None:
            stmt = stmt.where(Entity.kind == kind.value)
        return list(session.execute(stmt.order_by(Entity.clip_count.desc()).limit(limit)).scalars())

    @staticmethod
    def clips_for_entity(session: Session, entity_id: int, *, limit: int = 500) -> list[str]:
        return [
            str(row)
            for row in session.execute(
                select(ClipEntity.clip_id)
                .where(ClipEntity.entity_id == entity_id)
                .order_by(ClipEntity.confidence.desc())
                .limit(limit)
            ).scalars()
        ]

    @staticmethod
    def top(session: Session, kind: EntityKind, *, limit: int = 50) -> list[Entity]:
        return list(
            session.execute(
                select(Entity)
                .where(Entity.kind == kind.value)
                .order_by(Entity.clip_count.desc())
                .limit(limit)
            ).scalars()
        )


class EmbeddingRepository:
    @staticmethod
    def upsert(
        session: Session,
        clip_id: str,
        *,
        kind: str,
        model: str,
        dim: int,
        vector: bytes,
        norm: float,
    ) -> Embedding:
        row = session.execute(
            select(Embedding).where(
                Embedding.clip_id == clip_id,
                Embedding.kind == kind,
                Embedding.model == model,
            )
        ).scalar_one_or_none()
        if row is None:
            row = Embedding(clip_id=clip_id, kind=kind, model=model)
            session.add(row)
        row.dim = dim
        row.vector = vector
        row.norm = norm
        session.flush()
        return row

    @staticmethod
    def get(session: Session, clip_id: str, *, kind: str = "text") -> Optional[Embedding]:
        return session.execute(
            select(Embedding)
            .where(Embedding.clip_id == clip_id, Embedding.kind == kind)
            .limit(1)
        ).scalar_one_or_none()

    @staticmethod
    def iter_vectors(
        session: Session, *, kind: str = "text", model: Optional[str] = None, batch_size: int = 1000
    ) -> Iterator[Embedding]:
        stmt: Select[Any] = select(Embedding).where(Embedding.kind == kind)
        if model:
            stmt = stmt.where(Embedding.model == model)
        yield from session.execute(stmt.execution_options(yield_per=batch_size)).scalars()

    @staticmethod
    def count(session: Session) -> int:
        return int(session.execute(select(func.count()).select_from(Embedding)).scalar_one())


# ---------------------------------------------------------------------------
# Duplicates, events, runs, key-value
# ---------------------------------------------------------------------------


class DuplicateRepository:
    @staticmethod
    def group_for_clip(session: Session, clip_id: str) -> Optional[DuplicateGroup]:
        member = session.execute(
            select(DuplicateMember).where(DuplicateMember.clip_id == clip_id)
        ).scalar_one_or_none()
        if member is None:
            return None
        return session.get(DuplicateGroup, member.group_id)

    @staticmethod
    def record(
        session: Session,
        *,
        canonical_clip_id: str,
        members: Sequence[tuple[str, float]],
        method: str,
    ) -> DuplicateGroup:
        """Create or extend the duplicate group anchored on ``canonical_clip_id``."""
        group = session.execute(
            select(DuplicateGroup).where(
                DuplicateGroup.canonical_clip_id == canonical_clip_id
            )
        ).scalar_one_or_none()
        if group is None:
            group = DuplicateGroup(canonical_clip_id=canonical_clip_id, method=method)
            session.add(group)
            session.flush()
            session.add(
                DuplicateMember(
                    group_id=group.id,
                    clip_id=canonical_clip_id,
                    similarity=1.0,
                    method=method,
                    is_canonical=True,
                )
            )
        known = {
            str(row)
            for row in session.execute(
                select(DuplicateMember.clip_id).where(DuplicateMember.group_id == group.id)
            ).scalars()
        }
        for clip_id, similarity in members:
            if clip_id in known:
                continue
            session.add(
                DuplicateMember(
                    group_id=group.id, clip_id=clip_id, similarity=similarity, method=method
                )
            )
            known.add(clip_id)
        group.member_count = len(known)
        session.flush()
        return group

    @staticmethod
    def recent(session: Session, *, limit: int = 25) -> list[DuplicateGroup]:
        return list(
            session.execute(
                select(DuplicateGroup)
                .where(DuplicateGroup.member_count > 1)
                .order_by(DuplicateGroup.created_at.desc())
                .limit(limit)
            ).scalars()
        )

    @staticmethod
    def statistics(session: Session) -> dict[str, int]:
        groups = int(
            session.execute(
                select(func.count()).select_from(DuplicateGroup).where(DuplicateGroup.member_count > 1)
            ).scalar_one()
        )
        members = int(
            session.execute(select(func.count()).select_from(DuplicateMember)).scalar_one()
        )
        return {"duplicate_groups": groups, "duplicate_members": members}


class EventRepository:
    @staticmethod
    def log(
        session: Session,
        kind: str,
        message: str,
        *,
        level: str = "info",
        source_id: Optional[str] = None,
        clip_id: Optional[str] = None,
        job_id: Optional[int] = None,
        detail: Optional[dict[str, Any]] = None,
    ) -> None:
        session.add(
            Event(
                kind=kind,
                message=message[:4000],
                level=level,
                source_id=source_id,
                clip_id=clip_id,
                job_id=job_id,
                detail=detail,
            )
        )

    @staticmethod
    def recent(
        session: Session, *, limit: int = 50, level: Optional[str] = None
    ) -> list[Event]:
        stmt = select(Event).order_by(Event.created_at.desc()).limit(limit)
        if level:
            stmt = stmt.where(Event.level == level)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def purge(session: Session, *, keep: int = 20000) -> int:
        """Trim the event log, keeping the most recent ``keep`` rows."""
        threshold = session.execute(
            select(Event.id).order_by(Event.id.desc()).offset(keep).limit(1)
        ).scalar_one_or_none()
        if threshold is None:
            return 0
        return _rowcount(session.execute(delete(Event).where(Event.id < threshold)))


class RunRepository:
    @staticmethod
    def start(session: Session, *, mode: str, host: str, snapshot: dict[str, Any]) -> RunRecord:
        run = RunRecord(mode=mode, host=host, settings_snapshot=snapshot)
        session.add(run)
        session.flush()
        return run

    @staticmethod
    def finish(session: Session, run_id: int, **stats: Any) -> None:
        values = dict(stats)
        values["finished_at"] = utcnow()
        session.execute(update(RunRecord).where(RunRecord.id == run_id).values(**values))

    @staticmethod
    def latest(session: Session, *, limit: int = 10) -> list[RunRecord]:
        return list(
            session.execute(
                select(RunRecord).order_by(RunRecord.started_at.desc()).limit(limit)
            ).scalars()
        )


class KeyValueRepository:
    @staticmethod
    def get(session: Session, key: str) -> Optional[dict[str, Any]]:
        row = session.get(KeyValue, key)
        return row.value if row and isinstance(row.value, dict) else None

    @staticmethod
    def set(session: Session, key: str, value: dict[str, Any]) -> None:
        row = session.get(KeyValue, key)
        if row is None:
            session.add(KeyValue(key=key, value=value))
        else:
            row.value = value
        session.flush()


def _rowcount(result: Any) -> int:
    """Rows affected by a DML statement.

    ``rowcount`` is declared on ``CursorResult``, not on the generic ``Result``
    that ``Session.execute`` is typed as returning, so it is read defensively
    here rather than cast at a dozen call sites.
    """
    return int(getattr(result, "rowcount", 0) or 0)
