"""Crash-safe filesystem operations.

The pipeline can be killed at any instant -- power cut, ``Ctrl-C``, OOM. Every
file that lands in the Library must therefore appear atomically and completely,
or not at all. A half-written clip that looks finished is the one failure mode
that would silently poison the library, so all writes go through here.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional

from ..errors import StorageError

log = logging.getLogger(__name__)

#: Characters that are illegal or merely painful across macOS/Linux/Windows
#: and SMB shares. Library filenames are derived from clip titles, which come
#: from OCR and transcripts, so they can contain literally anything.
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")
_MAX_COMPONENT_BYTES = 200


def safe_filename(name: str, *, fallback: str = "clip", max_bytes: int = _MAX_COMPONENT_BYTES) -> str:
    """Turn arbitrary text into one safe path component.

    Truncation is byte-aware because most filesystems limit components to 255
    *bytes*, and an AI-generated title can be full of multi-byte characters.
    """
    cleaned = _UNSAFE_CHARS.sub(" ", name)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    encoded = cleaned.encode("utf-8")
    if len(encoded) > max_bytes:
        cleaned = encoded[:max_bytes].decode("utf-8", errors="ignore").rstrip()
    return cleaned or fallback


def unique_path(path: Path) -> Path:
    """Return ``path``, or ``path`` with a ``-2``, ``-3``... suffix if taken."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    for counter in range(2, 10_000):
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
    raise StorageError(f"could not find a free filename near {path}")


@contextlib.contextmanager
def atomic_write(target: Path, *, mode: str = "w", encoding: Optional[str] = "utf-8") -> Iterator[Any]:
    """Write to a temp file in the same directory, then rename into place.

    ``os.replace`` is atomic within a filesystem, so a reader either sees the
    old file or the complete new one.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, mode, encoding=encoding if "b" not in mode else None) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def write_json(target: Path, payload: Any, *, indent: int = 2) -> None:
    """Atomically write pretty-printed JSON (metadata sidecars, exports)."""
    with atomic_write(target) as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, sort_keys=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_copy(source: Path, target: Path) -> Path:
    """Copy via a temp file + rename, preserving mtime."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.partial"
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise StorageError(f"could not copy {source} -> {target}: {exc}") from exc
    return target


def atomic_move(source: Path, target: Path, *, overwrite: bool = False) -> Path:
    """Move a file, handling cross-device moves without leaving partials."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        target = unique_path(target)
    try:
        os.replace(source, target)
        return target
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise StorageError(f"could not move {source} -> {target}: {exc}") from exc
    atomic_copy(source, target)
    with contextlib.suppress(OSError):
        source.unlink()
    return target


def free_space_gb(path: Path) -> float:
    """Free space on the volume holding ``path``, in gibibytes."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return usage.free / (1024**3)


def directory_size_bytes(path: Path) -> int:
    """Recursive size of a directory, counting each inode once.

    Hardlinked duplicates are not double counted, so the dashboard's storage
    figure matches what ``du`` reports.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                stat = os.stat(os.path.join(root, name))
            except OSError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if stat.st_nlink > 1:
                if key in seen:
                    continue
                seen.add(key)
            total += stat.st_size
    return total
