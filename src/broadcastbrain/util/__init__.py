"""Dependency-free helpers shared across the pipeline."""

from __future__ import annotations

from .atomic import (
    atomic_copy,
    atomic_move,
    atomic_write,
    directory_size_bytes,
    free_space_gb,
    read_json,
    safe_filename,
    unique_path,
    write_json,
)
from .backoff import backoff_delay, retry_call
from .hashing import (
    decode_hash_sequence,
    encode_hash_sequence,
    hamming,
    quick_signature,
    sequence_distance,
    sha256_file,
    stable_id,
    to_signed64,
    to_unsigned64,
)
from .proc import CompletedCommand, require_tool, run, which
from .text import (
    canonical_key,
    clean_ocr,
    extract_copyright_year,
    extract_years,
    format_duration,
    format_timecode,
    normalize_whitespace,
    parse_timecode,
    slugify,
)

__all__ = [
    "CompletedCommand",
    "atomic_copy",
    "atomic_move",
    "atomic_write",
    "backoff_delay",
    "canonical_key",
    "clean_ocr",
    "decode_hash_sequence",
    "directory_size_bytes",
    "encode_hash_sequence",
    "extract_copyright_year",
    "extract_years",
    "format_duration",
    "format_timecode",
    "free_space_gb",
    "hamming",
    "normalize_whitespace",
    "parse_timecode",
    "quick_signature",
    "read_json",
    "require_tool",
    "retry_call",
    "run",
    "safe_filename",
    "sequence_distance",
    "sha256_file",
    "slugify",
    "stable_id",
    "to_signed64",
    "to_unsigned64",
    "unique_path",
    "which",
    "write_json",
]
