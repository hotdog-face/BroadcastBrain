"""SQLAlchemy 2.0 ORM models -- the single source of truth for the schema.

Design notes that matter for the life of this project:

* **Dialect neutrality.** Nothing here is SQLite-specific. JSON columns use
  ``JSONB`` on PostgreSQL via a variant; timestamps are timezone-aware; no
  ``AUTOINCREMENT`` tricks. Migrating is a matter of pointing
  ``database.url`` at PostgreSQL and re-running ``bb init``.
* **Stage tracking on the row.** Every source and clip records the last stage
  it completed. Resumption is then a query, not a bookkeeping file that can go
  stale.
* **Analyzer results are rows, not columns.** ``analyses`` is keyed by
  ``(clip_id, analyzer, analyzer_version)``. Adding an analyzer, or bumping one
  analyzer's version, re-runs exactly that analyzer on exactly the clips that
  need it -- never the whole library.
* **Nothing is deleted.** Duplicates, rejects and unknowns are all marked and
  kept.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

from ..types import (
    AnalysisStatus,
    Category,
    ClipStatus,
    EntityKind,
    JobState,
    SourceKind,
    SourceStatus,
    Stage,
)

#: JSON that becomes JSONB on PostgreSQL (indexable, binary, faster).
JsonType = JSON().with_variant(JSONB, "postgresql")


class UtcDateTime(TypeDecorator[datetime]):
    """A timestamp that is always timezone-aware UTC, on every backend.

    SQLite has no timezone-aware type: it silently drops the offset on write
    and hands back a naive datetime on read. That turns every Python-side
    comparison -- "has this lease expired?" -- into either a ``TypeError`` or,
    worse, a silently wrong answer once the two sides drift apart.

    Coercing here means the rest of the codebase can assume UTC-aware
    datetimes unconditionally, and the same code is correct on PostgreSQL
    where the offset really is stored.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Optional[datetime], dialect: Any) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


def utcnow() -> datetime:
    """Timezone-aware UTC now. Used as the default for every timestamp."""
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base with a shared type map."""

    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JsonType,
        list[Any]: JsonType,
    }


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, onupdate=utcnow, nullable=False
    )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class Source(Base, TimestampMixin):
    """One file dropped into Incoming (or fetched by a downloader).

    ``sha256`` is unique: re-dropping the same recording is recognized and
    skipped instantly, which is the foundation of "never reprocess".
    """

    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    mtime: Mapped[Optional[datetime]] = mapped_column(UtcDateTime())

    #: Cheap head/tail/size probe, computed before the expensive full hash.
    quick_signature: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True)

    kind: Mapped[str] = mapped_column(String(32), default=SourceKind.UNKNOWN.value, index=True)
    status: Mapped[str] = mapped_column(
        String(32), default=SourceStatus.DISCOVERED.value, nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(
        String(32), default=Stage.DISCOVERED.value, nullable=False, index=True
    )

    #: Full ffprobe output plus derived fields (see media.probe.MediaInfo).
    media_info: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)
    duration_s: Mapped[Optional[float]] = mapped_column(Float)
    container: Mapped[Optional[str]] = mapped_column(String(64))
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    fps: Mapped[Optional[float]] = mapped_column(Float)

    #: Where it came from: Internet Archive identifier, disc title number, etc.
    origin: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    origin_ref: Mapped[Optional[str]] = mapped_column(String(512), index=True)
    #: For disc images and multi-title rips: the parent source this was cut from.
    parent_source_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sources.id", ondelete="SET NULL"), index=True
    )

    clip_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    segmentation: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)
    error: Mapped[Optional[str]] = mapped_column(Text)
    reject_reason: Mapped[Optional[str]] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime())

    clips: Mapped[list[Clip]] = relationship(
        back_populates="source", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_sources_status_stage", "status", "stage"),
        Index("ix_sources_kind_status", "kind", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Source {self.filename!r} {self.status} {self.stage}>"


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------


class Clip(Base, TimestampMixin):
    """A logical unit of television: one spot, one episode, one bumper."""

    __tablename__ = "clips"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    source_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: Ordinal within the source; ``(source_id, seq)`` is unique and stable, so
    #: re-running segmentation is idempotent.
    seq: Mapped[int] = mapped_column(Integer, nullable=False)

    start_s: Mapped[float] = mapped_column(Float, nullable=False)
    end_s: Mapped[float] = mapped_column(Float, nullable=False)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False, index=True)

    #: Current location. ``processing_path`` is the working copy; ``file_path``
    #: is the final library location once materialized.
    processing_path: Mapped[Optional[str]] = mapped_column(Text)
    file_path: Mapped[Optional[str]] = mapped_column(Text, index=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text)
    transcript_path: Mapped[Optional[str]] = mapped_column(Text)
    metadata_path: Mapped[Optional[str]] = mapped_column(Text)
    size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)

    # -- technical ----------------------------------------------------------
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    aspect_ratio: Mapped[Optional[str]] = mapped_column(String(16))
    fps: Mapped[Optional[float]] = mapped_column(Float)
    video_codec: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    audio_codec: Mapped[Optional[str]] = mapped_column(String(32))
    video_bitrate: Mapped[Optional[int]] = mapped_column(Integer)
    audio_bitrate: Mapped[Optional[int]] = mapped_column(Integer)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer)
    channels: Mapped[Optional[int]] = mapped_column(Integer)
    interlaced: Mapped[Optional[bool]] = mapped_column(Boolean)

    # -- classification -----------------------------------------------------
    category: Mapped[str] = mapped_column(
        String(32), default=Category.UNKNOWN.value, nullable=False, index=True
    )
    category_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    #: Every vote that produced ``category`` -- classification is explainable.
    category_evidence: Mapped[Optional[list[Any]]] = mapped_column(JsonType)
    subcategory: Mapped[Optional[str]] = mapped_column(String(64), index=True)

    title: Mapped[Optional[str]] = mapped_column(String(512), index=True)
    series: Mapped[Optional[str]] = mapped_column(String(256), index=True)
    episode_title: Mapped[Optional[str]] = mapped_column(String(256))
    network: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    network_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    air_year: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    air_year_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    air_date: Mapped[Optional[datetime]] = mapped_column(UtcDateTime())
    genre: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    language: Mapped[Optional[str]] = mapped_column(String(16))

    # -- boolean flags the user explicitly asked to be able to filter on -----
    is_commercial: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    is_promo: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_episode: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_movie: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_music_video: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_intro: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    has_credits: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # -- derived text -------------------------------------------------------
    summary: Mapped[Optional[str]] = mapped_column(Text)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    quality_notes: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)

    # -- identity / dedupe --------------------------------------------------
    sha256: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    phash: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    phash_head: Mapped[Optional[int]] = mapped_column(BigInteger, index=True)
    chromaprint: Mapped[Optional[str]] = mapped_column(Text)
    duplicate_of: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("clips.id", ondelete="SET NULL"), index=True
    )
    is_alternate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    duplicate_similarity: Mapped[Optional[float]] = mapped_column(Float)

    # -- lifecycle ----------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(32), default=ClipStatus.NEW.value, nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(
        String(32), default=Stage.SEGMENTED.value, nullable=False, index=True
    )
    error: Mapped[Optional[str]] = mapped_column(Text)
    exported_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime())
    #: Operator override. When true, automated re-classification leaves it alone.
    locked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    source: Mapped[Source] = relationship(back_populates="clips")
    analyses: Mapped[list[Analysis]] = relationship(
        back_populates="clip", cascade="all, delete-orphan", passive_deletes=True
    )
    tags: Mapped[list[ClipTag]] = relationship(
        back_populates="clip", cascade="all, delete-orphan", passive_deletes=True
    )
    entity_links: Mapped[list[ClipEntity]] = relationship(
        back_populates="clip", cascade="all, delete-orphan", passive_deletes=True
    )
    texts: Mapped[list[ClipText]] = relationship(
        back_populates="clip", cascade="all, delete-orphan", passive_deletes=True
    )
    embeddings: Mapped[list[Embedding]] = relationship(
        back_populates="clip", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        UniqueConstraint("source_id", "seq", name="uq_clip_source_seq"),
        Index("ix_clips_category_year", "category", "air_year"),
        Index("ix_clips_network_year", "network", "air_year"),
        Index("ix_clips_status_stage", "status", "stage"),
        Index("ix_clips_dedupe_block", "duration_s", "phash_head"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Clip {self.id[:8]} {self.category} {self.duration_s:.1f}s>"


# ---------------------------------------------------------------------------
# Analyzer results
# ---------------------------------------------------------------------------


class Analysis(Base):
    """One analyzer's output for one clip.

    The ``(clip_id, analyzer)`` uniqueness plus a stored ``analyzer_version``
    is the whole caching strategy: the runner skips any analyzer whose stored
    version matches the current one and whose status is ``ok``.
    """

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clip_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    analyzer: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    analyzer_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default=AnalysisStatus.OK.value, nullable=False, index=True
    )
    backend: Mapped[Optional[str]] = mapped_column(String(64))
    payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    error: Mapped[Optional[str]] = mapped_column(Text)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )

    clip: Mapped[Clip] = relationship(back_populates="analyses")

    __table_args__ = (
        UniqueConstraint("clip_id", "analyzer", name="uq_analysis_clip_analyzer"),
        Index("ix_analyses_analyzer_status", "analyzer", "status"),
    )


class ClipText(Base):
    """Long text attached to a clip: transcript, OCR, or credits roll.

    Split out of ``clips`` so the hot table stays narrow -- a 90-minute
    transcript is hundreds of kilobytes and would otherwise be dragged into
    every listing query.
    """

    __tablename__ = "clip_texts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clip_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # transcript | ocr | credits
    language: Mapped[Optional[str]] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Word- or frame-level detail: transcript segments, OCR boxes + times.
    detail: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)
    confidence: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )

    clip: Mapped[Clip] = relationship(back_populates="texts")

    __table_args__ = (UniqueConstraint("clip_id", "kind", name="uq_cliptext_clip_kind"),)


class Entity(Base):
    """A person, brand, product, show, network, song or object.

    ``canonical_key`` is the aggressively normalized form, so "Taco Bell",
    "TACO BELL" and "taco bell" are one entity and one search hit.
    """

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    canonical_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    aliases: Mapped[Optional[list[Any]]] = mapped_column(JsonType)
    metadata_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)
    clip_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )

    links: Mapped[list[ClipEntity]] = relationship(
        back_populates="entity", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (UniqueConstraint("kind", "canonical_key", name="uq_entity_kind_key"),)

    @property
    def entity_kind(self) -> EntityKind:
        return EntityKind(self.kind)


class ClipEntity(Base):
    """Association between a clip and an entity, with provenance."""

    __tablename__ = "clip_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clip_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    #: Which analyzer asserted this, so a bad model can be undone selectively.
    source_analyzer: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Seconds into the clip where the entity was observed, when known.
    at_s: Mapped[Optional[float]] = mapped_column(Float)

    clip: Mapped[Clip] = relationship(back_populates="entity_links")
    entity: Mapped[Entity] = relationship(back_populates="links")

    __table_args__ = (
        UniqueConstraint("clip_id", "entity_id", "source_analyzer", name="uq_clip_entity_src"),
    )


class ClipTag(Base):
    """Free-form searchable tag."""

    __tablename__ = "clip_tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clip_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tag: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="auto", nullable=False)

    clip: Mapped[Clip] = relationship(back_populates="tags")

    __table_args__ = (UniqueConstraint("clip_id", "tag", name="uq_cliptag_clip_tag"),)


class Embedding(Base):
    """A dense vector for semantic search.

    Stored as a raw float32 BLOB rather than JSON: 384 floats is 1.5 KB packed
    versus ~9 KB as text, and at 100k clips that difference is the difference
    between an index that fits in page cache and one that does not.
    """

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clip_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), default="text", nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    #: L2 norm, stored so cosine similarity is a dot product at query time.
    norm: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )

    clip: Mapped[Clip] = relationship(back_populates="embeddings")

    __table_args__ = (UniqueConstraint("clip_id", "kind", "model", name="uq_embedding_clip_kind"),)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class DuplicateGroup(Base):
    """A set of clips that are the same content; one is canonical."""

    __tablename__ = "duplicate_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    canonical_clip_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )

    members: Mapped[list[DuplicateMember]] = relationship(
        back_populates="group", cascade="all, delete-orphan", passive_deletes=True
    )


class DuplicateMember(Base):
    __tablename__ = "duplicate_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("duplicate_groups.id", ondelete="CASCADE"), nullable=False, index=True
    )
    clip_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("clips.id", ondelete="CASCADE"), nullable=False, index=True
    )
    similarity: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    is_canonical: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    group: Mapped[DuplicateGroup] = relationship(back_populates="members")

    __table_args__ = (UniqueConstraint("group_id", "clip_id", name="uq_dupmember_group_clip"),)


# ---------------------------------------------------------------------------
# Queue, downloads, events
# ---------------------------------------------------------------------------


class Job(Base, TimestampMixin):
    """A durable unit of work.

    Durability is the point: the queue lives in the database, so ``Ctrl-C``,
    a kernel panic or a power cut costs at most the jobs that were mid-flight,
    and those are reclaimed by lease expiry on the next run.
    """

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    #: Stable hash of the work being requested. Unique, so enqueueing the same
    #: job twice is a no-op instead of duplicate processing.
    dedupe_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType, nullable=False, default=dict)
    state: Mapped[str] = mapped_column(
        String(16), default=JobState.PENDING.value, nullable=False, index=True
    )
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    #: Set while running; a worker that dies leaves an expired lease behind,
    #: which the next run reclaims automatically.
    lease_expires_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime(), index=True)
    worker_id: Mapped[Optional[str]] = mapped_column(String(64))
    #: Do not run before this time (retry backoff).
    available_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False, index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime())
    finished_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime())
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    #: Requires a GPU slot. Lets a 1-GPU box run 12 CPU workers safely.
    needs_gpu: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    progress_note: Mapped[Optional[str]] = mapped_column(String(256))

    __table_args__ = (
        Index("ix_jobs_claim", "state", "available_at", "priority"),
        Index("ix_jobs_kind_state", "kind", "state"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Job {self.id} {self.kind} {self.state}>"


class Download(Base, TimestampMixin):
    """An item fetched (or to be fetched) from an external archive source."""

    __tablename__ = "downloads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    identifier: Mapped[str] = mapped_column(String(512), nullable=False)
    file_name: Mapped[Optional[str]] = mapped_column(String(512))
    url: Mapped[Optional[str]] = mapped_column(Text)
    title: Mapped[Optional[str]] = mapped_column(Text)
    query: Mapped[Optional[str]] = mapped_column(Text)
    state: Mapped[str] = mapped_column(String(24), default="pending", nullable=False, index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    downloaded_bytes: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    local_path: Mapped[Optional[str]] = mapped_column(Text)
    sha256: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    item_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)
    error: Mapped[Optional[str]] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime())

    __table_args__ = (
        UniqueConstraint("source_name", "identifier", "file_name", name="uq_download_item_file"),
    )


class Event(Base):
    """Append-only activity log powering the dashboard and nightly reports."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False, index=True
    )
    level: Mapped[str] = mapped_column(String(16), default="info", nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    clip_id: Mapped[Optional[str]] = mapped_column(String(36), index=True)
    job_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    detail: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)


class RunRecord(Base):
    """One invocation of the pipeline, for the morning-after report."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, nullable=False
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(UtcDateTime())
    mode: Mapped[str] = mapped_column(String(32), default="run", nullable=False)
    host: Mapped[Optional[str]] = mapped_column(String(128))
    sources_seen: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sources_completed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    clips_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    clips_duplicated: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    jobs_completed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    jobs_failed: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    report_path: Mapped[Optional[str]] = mapped_column(Text)
    settings_snapshot: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)


class KeyValue(Base):
    """Small persistent state: schema version, last scan time, index cursors."""

    __tablename__ = "kv"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType)
    updated_at: Mapped[datetime] = mapped_column(
        UtcDateTime(), default=utcnow, onupdate=utcnow, nullable=False
    )


#: Tables the FTS index mirrors. Used by the search backend when rebuilding.
SEARCHABLE_TABLES = (Clip.__tablename__, ClipText.__tablename__, ClipTag.__tablename__)
