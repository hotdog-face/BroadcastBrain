"""Semantic search over stored embeddings.

Brute force, deliberately. At a hundred thousand clips a 384-dimension scan is
a few hundred milliseconds of pure arithmetic with NumPy and a couple of
seconds without -- fast enough for an interactive dashboard, and free of the
operational burden that an approximate index brings (build times, staleness,
another file to corrupt).

If the library ever outgrows that, this module is the only place that changes:
add ``sqlite-vec`` or FAISS behind :func:`search` and nothing else moves.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from ..analyze.embed import cosine, load_embedding_model, unpack_vector
from ..config import Settings
from ..db.base import Database
from ..db.models import Embedding
from ..db.repo import EmbeddingRepository
from ..errors import BackendUnavailableError

log = logging.getLogger(__name__)

#: Beyond this many stored vectors, warn once that an index would help.
BRUTE_FORCE_WARN_THRESHOLD = 250_000


def _numpy() -> Optional[Any]:
    try:
        import numpy

        return numpy
    except ImportError:
        return None


def embed_query(settings: Settings, query: str) -> list[float]:
    """Embed a search string with the same model the clips used.

    Raises:
        BackendUnavailableError: sentence-transformers is not installed, so
            semantic search is simply unavailable -- keyword search still works.
    """
    from .. import capabilities

    if not capabilities.has("sentence_transformers"):
        raise BackendUnavailableError(
            "search", "semantic search needs: pip install 'broadcastbrain[embeddings]'"
        )
    model = load_embedding_model(settings.analysis.embedding_model)
    vector = model.encode(query, normalize_embeddings=True, show_progress_bar=False)
    return [float(v) for v in vector]


def search(
    db: Database,
    query_vector: Sequence[float],
    *,
    limit: int = 50,
    min_score: float = 0.0,
    kind: str = "text",
    exclude_clip_id: Optional[str] = None,
) -> list[tuple[str, float]]:
    """Rank clips by cosine similarity to ``query_vector``."""
    if not query_vector:
        return []

    numpy = _numpy()
    with db.session() as session:
        rows = list(EmbeddingRepository.iter_vectors(session, kind=kind))
        if len(rows) > BRUTE_FORCE_WARN_THRESHOLD:
            log.warning(
                "%d vectors in a brute-force scan; consider an ANN index", len(rows)
            )
        scored = (
            _score_numpy(numpy, rows, query_vector)
            if numpy is not None
            else _score_python(rows, query_vector)
        )

    results = [
        (clip_id, score)
        for clip_id, score in scored
        if score >= min_score and clip_id != exclude_clip_id
    ]
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:limit]


def _score_numpy(
    numpy: Any, rows: Sequence[Embedding], query_vector: Sequence[float]
) -> list[tuple[str, float]]:
    """Vectorized scoring: one matrix multiply for the whole library."""
    if not rows:
        return []
    dim = rows[0].dim
    usable = [row for row in rows if row.dim == dim]
    matrix = numpy.frombuffer(b"".join(row.vector for row in usable), dtype=numpy.float32)
    matrix = matrix.reshape(len(usable), dim)
    query = numpy.asarray(query_vector, dtype=numpy.float32)

    # Vectors are stored normalized, but renormalizing costs nothing and makes
    # this correct even for embeddings written by a future analyzer that is not.
    norms = numpy.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    query_norm = float(numpy.linalg.norm(query)) or 1.0
    scores = (matrix @ query) / (norms * query_norm)
    return [(row.clip_id, float(score)) for row, score in zip(usable, scores)]


def _score_python(
    rows: Sequence[Embedding], query_vector: Sequence[float]
) -> list[tuple[str, float]]:
    """Pure-Python fallback so semantic search works without NumPy."""
    return [
        (row.clip_id, cosine(unpack_vector(row.vector), query_vector))
        for row in rows
        if row.dim == len(query_vector)
    ]


def similar_to_clip(
    db: Database, clip_id: str, *, limit: int = 20, min_score: float = 0.55
) -> list[tuple[str, float]]:
    """Find clips like a given clip -- "more like this" in the dashboard."""
    with db.session() as session:
        embedding = EmbeddingRepository.get(session, clip_id)
        if embedding is None:
            return []
        vector = unpack_vector(embedding.vector)
    return search(db, vector, limit=limit, min_score=min_score, exclude_clip_id=clip_id)


def coverage(db: Database) -> dict[str, int]:
    """How much of the library is semantically searchable."""
    from sqlalchemy import func, select

    from ..db.models import Clip

    with db.session() as session:
        clips = int(session.execute(select(func.count()).select_from(Clip)).scalar_one())
        embedded = EmbeddingRepository.count(session)
    return {"clips": clips, "embedded": embedded, "missing": max(0, clips - embedded)}
