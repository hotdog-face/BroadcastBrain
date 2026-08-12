"""Optional archive downloaders.

Importing this package registers the built-in sources. Adding another archive
means writing one module that subclasses :class:`ArchiveSource` and importing
it here -- the download manager, queue and pipeline need no changes.
"""

from __future__ import annotations

from .base import (
    ArchiveSource,
    RemoteFile,
    RemoteItem,
    SearchSpec,
    all_sources,
    get_source,
    register_source,
    source_names,
)
from .internet_archive import ERA_PRESETS, InternetArchiveSource
from .manager import DownloadManager, FetchPlan

__all__ = [
    "ERA_PRESETS",
    "ArchiveSource",
    "DownloadManager",
    "FetchPlan",
    "InternetArchiveSource",
    "RemoteFile",
    "RemoteItem",
    "SearchSpec",
    "all_sources",
    "get_source",
    "register_source",
    "source_names",
]
