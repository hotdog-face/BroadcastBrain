"""Search: structured filters, keyword index and semantic vectors."""

from __future__ import annotations

from .fts import Fts5Backend, KeywordBackend, LikeBackend, get_backend, index_clips
from .query import ParsedQuery, parse, to_fts_expression
from .service import SearchHit, SearchResponse, SearchService
from .vectors import coverage, embed_query, similar_to_clip

__all__ = [
    "Fts5Backend",
    "KeywordBackend",
    "LikeBackend",
    "ParsedQuery",
    "SearchHit",
    "SearchResponse",
    "SearchService",
    "coverage",
    "embed_query",
    "get_backend",
    "index_clips",
    "parse",
    "similar_to_clip",
    "to_fts_expression",
]
