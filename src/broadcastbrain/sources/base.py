"""The archive source contract.

An archive source is anything that can be searched for recordings and can hand
back downloadable files. Internet Archive is the one implemented; the protocol
exists so that adding another (a private NAS index, YouTube-dl-backed sources,
a friend's FTP) is a new module and a registry entry, with no changes to the
downloader, the queue, or the pipeline.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class RemoteFile:
    """One downloadable file within an item."""

    name: str
    url: str
    size_bytes: int = 0
    format: str = ""
    checksum: Optional[str] = None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


@dataclass
class RemoteItem:
    """A searchable unit in an archive: usually one recording or one upload."""

    identifier: str
    title: str
    source_name: str
    description: str = ""
    year: Optional[int] = None
    date: Optional[str] = None
    collection: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    files: list[RemoteFile] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    def best_file(self, accept_formats: list[str]) -> Optional[RemoteFile]:
        """Pick the most useful file to download.

        Preference order is by the operator's configured format list, then by
        size -- the largest acceptable file is almost always the highest
        quality copy, which is what deduplication will end up keeping anyway.
        """
        if not self.files:
            return None
        ranked: list[tuple[int, int, RemoteFile]] = []
        for file in self.files:
            try:
                priority = accept_formats.index(file.format)
            except ValueError:
                priority = len(accept_formats)
            ranked.append((priority, -file.size_bytes, file))
        ranked.sort(key=lambda entry: (entry[0], entry[1]))
        best = ranked[0][2]
        # Everything was an unacceptable format.
        if accept_formats and ranked[0][0] == len(accept_formats):
            return None
        return best

    def as_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "title": self.title,
            "source": self.source_name,
            "year": self.year,
            "date": self.date,
            "collection": self.collection,
            "subjects": self.subjects,
            "file_count": len(self.files),
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class SearchSpec:
    """A search request, normalized across sources."""

    query: str
    limit: int = 25
    collections: list[str] = field(default_factory=list)
    date_range: Optional[str] = None
    media_type: str = "movies"


class ArchiveSource(abc.ABC):
    """Base class for a downloadable archive."""

    #: Stable identifier stored on every download row.
    name: str = "archive"
    description: str = ""

    @abc.abstractmethod
    def search(self, spec: SearchSpec) -> Iterator[RemoteItem]:
        """Yield matching items, newest or most relevant first."""

    @abc.abstractmethod
    def fetch_item(self, identifier: str) -> Optional[RemoteItem]:
        """Full metadata, including the file list, for one item."""

    @abc.abstractmethod
    def download(
        self,
        item: RemoteItem,
        file: RemoteFile,
        destination: Path,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> Path:
        """Download one file to ``destination`` and return the final path.

        Implementations must download to a temporary name and move into place,
        so a partial download is never mistaken for a complete one.
        """

    def available(self) -> bool:
        """Whether this source can be used right now."""
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "available": self.available(),
        }


_REGISTRY: dict[str, type[ArchiveSource]] = {}


def register_source(cls: type[ArchiveSource]) -> type[ArchiveSource]:
    """Class decorator that makes a source available to the downloader."""
    _REGISTRY[cls.name] = cls
    return cls


def get_source(name: str) -> Optional[ArchiveSource]:
    cls = _REGISTRY.get(name)
    return cls() if cls is not None else None


def all_sources() -> list[ArchiveSource]:
    return [cls() for cls in _REGISTRY.values()]


def source_names() -> list[str]:
    return sorted(_REGISTRY)
