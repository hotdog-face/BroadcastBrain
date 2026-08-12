"""The library: on-disk layout and clip materialization."""

from __future__ import annotations

from .layout import (
    CATEGORY_DIRECTORIES,
    LIBRARY_DIRECTORIES,
    category_directory,
    clip_filename,
    clip_relative_path,
    ensure_library_tree,
    export_slug,
)
from .materialize import Materializer, MaterializeResult, build_metadata_document

__all__ = [
    "CATEGORY_DIRECTORIES",
    "LIBRARY_DIRECTORIES",
    "MaterializeResult",
    "Materializer",
    "build_metadata_document",
    "category_directory",
    "clip_filename",
    "clip_relative_path",
    "ensure_library_tree",
    "export_slug",
]
