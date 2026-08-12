"""Semantic embeddings for "find clips like this one".

A single vector per clip, computed from the text that describes it: title,
summary, transcript, on-screen text, tags and entities. Keyword search finds
"Taco Bell"; embeddings find "that fast food ad with the talking chihuahua"
when you cannot remember the brand.

Vectors are stored packed as float32 with their norm precomputed, so cosine
similarity at query time is a dot product over a contiguous buffer -- fast
enough to brute-force a hundred thousand clips without an ANN index. If the
library grows past that, :mod:`broadcastbrain.search.vectors` is the one place
that would need an index.
"""

from __future__ import annotations

import array
import logging
import math
import threading
from typing import Any, Optional, Sequence

from ..errors import AnalyzerError
from ..util.text import truncate
from .base import AnalysisContext, Analyzer, AnalyzerResult
from .registry import register

log = logging.getLogger(__name__)

_model_lock = threading.Lock()
_models: dict[str, Any] = {}

#: Sentence-transformer models truncate anyway; this bounds tokenizer cost on
#: a feature-length transcript.
MAX_EMBED_CHARS = 4000


def load_embedding_model(model_name: str) -> Any:
    """Load and cache a sentence-transformers model for this process."""
    with _model_lock:
        model = _models.get(model_name)
        if model is not None:
            return model
        from sentence_transformers import SentenceTransformer

        from .. import capabilities

        device = capabilities.torch_device("auto")
        log.info("loading embedding model %s on %s", model_name, device)
        model = SentenceTransformer(model_name, device=device)
        _models[model_name] = model
        return model


def pack_vector(values: Sequence[float]) -> tuple[bytes, float, int]:
    """Pack floats into a float32 BLOB, returning ``(blob, norm, dim)``."""
    packed = array.array("f", [float(v) for v in values])
    norm = math.sqrt(sum(v * v for v in packed)) or 1.0
    return packed.tobytes(), norm, len(packed)


def unpack_vector(blob: bytes) -> list[float]:
    """Inverse of :func:`pack_vector`."""
    values = array.array("f")
    values.frombytes(blob)
    return list(values)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, safe against zero vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_document(
    *,
    title: Optional[str],
    summary: Optional[str],
    transcript: Optional[str],
    ocr_text: Optional[str],
    tags: Sequence[str] = (),
    entities: Sequence[str] = (),
    network: Optional[str] = None,
    year: Optional[int] = None,
    category: Optional[str] = None,
) -> str:
    """Assemble the text that represents a clip in embedding space.

    Field order is not arbitrary: the most identifying text goes first because
    the model truncates from the end, so a long transcript can never push the
    title and brand out of the window.
    """
    parts = [
        title or "",
        summary or "",
        f"network: {network}" if network else "",
        f"year: {year}" if year else "",
        f"type: {category}" if category else "",
        ("tags: " + ", ".join(tags)) if tags else "",
        ("featuring: " + ", ".join(entities)) if entities else "",
        ("on screen: " + ocr_text) if ocr_text else "",
        transcript or "",
    ]
    return truncate("\n".join(p for p in parts if p), MAX_EMBED_CHARS)


@register
class EmbeddingAnalyzer(Analyzer):
    """Compute the semantic vector for a clip."""

    name = "embedding"
    version = "1"
    requires = ("sentence_transformers",)
    needs_gpu = True
    order = 90
    description = "Sentence-transformer embedding for semantic search"

    def run(self, ctx: AnalysisContext) -> AnalyzerResult:
        self.check_backends()
        cfg = ctx.settings.analysis

        summary_payload = ctx.payload_of("summary")
        estimation = ctx.payload_of("estimation")
        document = build_document(
            title=summary_payload.get("title"),
            summary=summary_payload.get("summary"),
            transcript=ctx.text_of("speech", "text"),
            ocr_text=ctx.text_of("ocr", "text"),
            network=_estimate(estimation, "network"),
            year=_estimate(estimation, "air_year"),
        )
        if not document.strip():
            # An embedding of nothing is noise that would pollute similarity
            # search; skipping is the honest outcome.
            from ..errors import BackendUnavailableError

            raise BackendUnavailableError(self.name, "no text available to embed yet")

        try:
            model = load_embedding_model(cfg.embedding_model)
            vector = model.encode(document, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:
            raise AnalyzerError(self.name, f"embedding failed: {exc}") from exc

        values = [float(v) for v in vector]
        blob, norm, dim = pack_vector(values)

        result = AnalyzerResult(backend=f"sentence-transformers/{cfg.embedding_model}")
        result.payload = {
            "model": cfg.embedding_model,
            "dim": dim,
            "document_chars": len(document),
            # The vector itself lives in the embeddings table, not here.
            "stored": True,
        }
        result.confidence = 1.0
        # Consumed by the runner, which writes it to the embeddings table.
        result.fields["__embedding__"] = {
            "kind": "text",
            "model": cfg.embedding_model,
            "dim": dim,
            "vector": blob,
            "norm": norm,
        }
        return result


def _estimate(payload: dict[str, Any], key: str) -> Optional[Any]:
    entry = payload.get(key)
    if isinstance(entry, dict):
        return entry.get("value")
    return None


def clear_model_cache() -> None:
    with _model_lock:
        _models.clear()
