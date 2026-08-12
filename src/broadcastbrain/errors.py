"""Exception hierarchy.

The distinction that matters operationally is :class:`RetryableError` vs
:class:`PermanentError`. The queue retries the former with backoff and sends
the latter straight to `dead` so a worker never burns three attempts on a file
that will never decode.
"""

from __future__ import annotations

from typing import Optional


class BroadcastBrainError(Exception):
    """Base class for everything this package raises deliberately."""


class ConfigError(BroadcastBrainError):
    """Configuration is missing, malformed, or self-contradictory."""


class RetryableError(BroadcastBrainError):
    """Transient failure: disk full, network blip, GPU busy, lock contention."""

    def __init__(self, message: str, *, retry_after_s: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class PermanentError(BroadcastBrainError):
    """The work item cannot succeed no matter how many times it is retried."""


class MediaError(BroadcastBrainError):
    """Base for media-handling failures."""


class ProbeError(MediaError, PermanentError):
    """ffprobe could not make sense of the file."""


class UnsupportedMediaError(MediaError, PermanentError):
    """Recognized container, but nothing we can decode (e.g. audio-only, DRM)."""


class ToolNotFoundError(MediaError, RetryableError):
    """A required external binary (ffmpeg, ffprobe, fpcalc) is not installed.

    Retryable on purpose: install the tool and resume; no work is lost.
    """


class SubprocessError(MediaError):
    """An external tool exited non-zero."""

    def __init__(self, command: str, returncode: int, stderr: str) -> None:
        super().__init__(f"{command} exited {returncode}: {stderr[-2000:]}")
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


class SubprocessTimeoutError(SubprocessError, RetryableError):
    """An external tool exceeded its wall-clock budget and was killed."""

    def __init__(self, command: str, timeout_s: float) -> None:
        SubprocessError.__init__(self, command, -1, f"timed out after {timeout_s}s")
        self.timeout_s = timeout_s


class SegmentationError(BroadcastBrainError):
    """Signals were collected but no sane segmentation could be derived."""


class AnalyzerError(BroadcastBrainError):
    """An analyzer failed in a way that is specific to that analyzer.

    Never fatal to a clip: the runner records the failure against that one
    analyzer and the rest of the pipeline continues.
    """

    def __init__(self, analyzer: str, message: str) -> None:
        super().__init__(f"[{analyzer}] {message}")
        self.analyzer = analyzer


class BackendUnavailableError(AnalyzerError):
    """The optional dependency backing an analyzer is not installed.

    Recorded as `skipped`, not `failed`, so `bb reanalyze --missing` picks it up
    once the backend appears.
    """


class StorageError(BroadcastBrainError):
    """Filesystem operation failed (permissions, cross-device, out of space)."""


class DatabaseError(BroadcastBrainError):
    """Schema or integrity problem that the application must not paper over."""


class ExportError(BroadcastBrainError):
    """Fieldstation export could not be produced."""


class DownloadError(RetryableError):
    """An archive source failed to deliver an item."""
