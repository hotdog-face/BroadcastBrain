"""Internet Archive source.

Implemented against the public HTTP endpoints with ``httpx`` rather than the
``internetarchive`` package. That keeps the dependency optional, keeps the
failure modes visible (a 503 is a 503), and lets downloads stream to disk with
resume support -- which matters when a "broadcast day" upload is 12 GB and the
connection drops at hour three.

Search results are ranked by the archive; this module's job is to filter them
down to things that are actually usable: right media type, plausible size, a
format we can decode, and not something we have already got.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import httpx

from ..constants import (
    IA_DOWNLOAD_TIMEOUT_S,
    IA_DOWNLOAD_URL,
    IA_MAX_RESULTS_PER_QUERY,
    IA_METADATA_URL,
    IA_PAGE_SIZE,
    IA_REQUEST_TIMEOUT_S,
    IA_RETRY_ATTEMPTS,
    IA_SEARCH_URL,
    IA_USER_AGENT,
)
from ..errors import DownloadError
from ..util.atomic import atomic_move
from ..util.backoff import backoff_delay
from .base import (
    ArchiveSource,
    ProgressCallback,
    RemoteFile,
    RemoteItem,
    SearchSpec,
    register_source,
)

log = logging.getLogger(__name__)

#: Fields the search endpoint should return.
_SEARCH_FIELDS = (
    "identifier", "title", "description", "year", "date", "collection",
    "subject", "mediatype", "item_size", "downloads",
)

#: Formats worth downloading, mapped to a sensible extension when the item's
#: file list does not carry one.
_FORMAT_EXTENSIONS = {
    "MPEG4": ".mp4",
    "h.264": ".mp4",
    "512Kb MPEG4": ".mp4",
    "HiRes MPEG4": ".mp4",
    "MPEG2": ".mpg",
    "Ogg Video": ".ogv",
    "Matroska": ".mkv",
    "DivX": ".avi",
}

#: Chunk size for streaming downloads. Large enough to be efficient, small
#: enough that progress reporting feels live.
_DOWNLOAD_CHUNK = 4 * 1024 * 1024


@register_source
class InternetArchiveSource(ArchiveSource):
    """Search and download from archive.org."""

    name = "internet_archive"
    description = "archive.org public collections (TV, commercials, broadcast recordings)"

    def __init__(self, *, client: Optional[httpx.Client] = None) -> None:
        self._client = client
        self._owns_client = client is None

    # -- client -------------------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                headers={"User-Agent": IA_USER_AGENT},
                timeout=IA_REQUEST_TIMEOUT_S,
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    # -- search -------------------------------------------------------------

    def search(self, spec: SearchSpec) -> Iterator[RemoteItem]:
        """Page through advancedsearch results, yielding usable items."""
        yielded = 0
        page = 1
        query = _build_query(spec)
        log.info("archive.org search: %s", query)

        while yielded < min(spec.limit, IA_MAX_RESULTS_PER_QUERY):
            payload = self._request_json(
                IA_SEARCH_URL,
                params={
                    "q": query,
                    "fl[]": list(_SEARCH_FIELDS),
                    "rows": min(IA_PAGE_SIZE, spec.limit - yielded),
                    "page": page,
                    "output": "json",
                    "sort[]": "downloads desc",
                },
            )
            docs = ((payload or {}).get("response") or {}).get("docs") or []
            if not docs:
                break

            for doc in docs:
                item = _item_from_doc(doc, self.name)
                if item is None:
                    continue
                yield item
                yielded += 1
                if yielded >= spec.limit:
                    return
            page += 1

    def fetch_item(self, identifier: str) -> Optional[RemoteItem]:
        """Fetch an item's full metadata, including its file list."""
        payload = self._request_json(f"{IA_METADATA_URL}/{identifier}")
        if not payload or not payload.get("metadata"):
            return None

        metadata = payload["metadata"]
        item = RemoteItem(
            identifier=identifier,
            title=str(metadata.get("title") or identifier),
            source_name=self.name,
            description=_first_string(metadata.get("description")),
            year=_as_year(metadata.get("year") or metadata.get("date")),
            date=_first_string(metadata.get("date")),
            collection=_as_list(metadata.get("collection")),
            subjects=_as_list(metadata.get("subject")),
            metadata=metadata,
        )
        server = payload.get("server") or "archive.org"
        directory = payload.get("dir") or f"/{identifier}"

        for entry in payload.get("files") or []:
            file_format = str(entry.get("format") or "")
            if file_format not in _FORMAT_EXTENSIONS:
                continue
            name = str(entry.get("name") or "")
            if not name:
                continue
            item.files.append(
                RemoteFile(
                    name=name,
                    url=f"https://{server}{directory}/{name}",
                    size_bytes=int(entry.get("size") or 0),
                    format=file_format,
                    checksum=entry.get("sha1"),
                )
            )
        return item

    # -- download -----------------------------------------------------------

    def download(
        self,
        item: RemoteItem,
        file: RemoteFile,
        destination: Path,
        *,
        progress: Optional[ProgressCallback] = None,
    ) -> Path:
        """Stream a file to disk, resuming a partial download if one exists."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        already = partial.stat().st_size if partial.exists() else 0

        headers: dict[str, str] = {}
        if already > 0:
            # Resume where the last attempt stopped rather than starting a 12 GB
            # download again from zero.
            headers["Range"] = f"bytes={already}-"
            log.info("resuming %s at %.1f MB", file.name, already / (1024 * 1024))

        url = file.url or f"{IA_DOWNLOAD_URL}/{item.identifier}/{file.name}"
        started = time.monotonic()
        downloaded = already

        try:
            with self.client.stream(
                "GET", url, headers=headers, timeout=IA_DOWNLOAD_TIMEOUT_S
            ) as response:
                if response.status_code == 416:  # already complete
                    response.close()
                    return atomic_move(partial, destination, overwrite=True)
                response.raise_for_status()

                total = file.size_bytes or int(response.headers.get("content-length") or 0) + already
                mode = "ab" if already and response.status_code == 206 else "wb"
                if mode == "wb":
                    downloaded = 0

                with partial.open(mode) as handle:
                    for chunk in response.iter_bytes(_DOWNLOAD_CHUNK):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if progress and total:
                            progress(
                                min(1.0, downloaded / total),
                                f"{file.name}: {downloaded / (1024**2):.0f}/"
                                f"{total / (1024**2):.0f} MB",
                            )
        except httpx.HTTPError as exc:
            raise DownloadError(f"download failed for {item.identifier}/{file.name}: {exc}") from exc

        elapsed = max(0.001, time.monotonic() - started)
        log.info(
            "downloaded %s (%.1f MB in %.0fs, %.1f MB/s)",
            file.name, downloaded / (1024**2), elapsed, downloaded / (1024**2) / elapsed,
        )
        return atomic_move(partial, destination, overwrite=True)

    # -- helpers ------------------------------------------------------------

    def _request_json(self, url: str, *, params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        """GET JSON with retries. archive.org rate-limits and occasionally 503s."""
        last_error: Optional[Exception] = None
        for attempt in range(1, IA_RETRY_ATTEMPTS + 1):
            try:
                response = self.client.get(url, params=params)
                if response.status_code in (429, 503):
                    raise httpx.HTTPStatusError(
                        f"archive.org returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                return payload if isinstance(payload, dict) else None
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt >= IA_RETRY_ATTEMPTS:
                    break
                delay = backoff_delay(attempt)
                log.warning(
                    "archive.org request failed (%s); retrying in %.0fs", exc, delay
                )
                time.sleep(delay)
        log.error("archive.org request gave up: %s", last_error)
        return None


# ---------------------------------------------------------------------------
# Query construction and parsing
# ---------------------------------------------------------------------------


def _build_query(spec: SearchSpec) -> str:
    """Compose a Lucene query for advancedsearch.php."""
    parts = [f"({spec.query})"] if spec.query else []
    parts.append(f'mediatype:"{spec.media_type}"')
    if spec.collections:
        joined = " OR ".join(f'collection:"{c}"' for c in spec.collections)
        parts.append(f"({joined})")
    if spec.date_range:
        parts.append(f"date:{spec.date_range}")
    return " AND ".join(parts)


def _item_from_doc(doc: dict[str, Any], source_name: str) -> Optional[RemoteItem]:
    identifier = doc.get("identifier")
    if not identifier:
        return None
    return RemoteItem(
        identifier=str(identifier),
        title=str(doc.get("title") or identifier),
        source_name=source_name,
        description=_first_string(doc.get("description")),
        year=_as_year(doc.get("year") or doc.get("date")),
        date=_first_string(doc.get("date")),
        collection=_as_list(doc.get("collection")),
        subjects=_as_list(doc.get("subject")),
        metadata={"item_size": doc.get("item_size"), "downloads": doc.get("downloads")},
    )


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _first_string(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _as_year(value: Any) -> Optional[int]:
    text = _first_string(value)
    if not text:
        return None
    import re

    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else None


#: Ready-made searches for the era this project targets. ``bb fetch --presets``
#: runs these; they are also the documented example of what a query looks like.
ERA_PRESETS: dict[str, str] = {
    "mtv-2004": 'MTV 2004',
    "vh1": 'VH1 "music television"',
    "discovery": '"Discovery Channel" recording',
    "adult-swim": '"adult swim" cartoon network',
    "comedy-central": '"Comedy Central" recorded',
    "nickelodeon": 'Nickelodeon "nick at nite"',
    "commercials": 'television commercials 2000s',
    "off-tv": '"recorded off tv"',
    "cable-recording": '"cable recording" OR "cable tv recording"',
    "broadcast-day": '"broadcast day"',
}
