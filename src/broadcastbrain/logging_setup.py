"""Logging configuration.

One rotating file (everything, DEBUG) plus a readable console stream. Optional
JSON output for anyone who wants to ship logs somewhere. Worker threads tag
their records with the job they are running, which is what makes an overnight
log readable the next morning.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from .config import Settings
from .constants import APP_SLUG, LOG_FILENAME

_configured = False
_lock = threading.Lock()

#: Thread-local context merged into every record emitted by that thread.
_context = threading.local()


class ContextFilter(logging.Filter):
    """Attach thread-local job context (job id, clip id, source) to records."""

    def filter(self, record: logging.LogRecord) -> bool:
        data = getattr(_context, "data", None) or {}
        record.context = data
        record.context_str = (
            " ".join(f"{k}={v}" for k, v in data.items()) if data else ""
        )
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "thread": record.threadName,
        }
        context = getattr(record, "context", None)
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Compact, aligned console output with the job context appended."""

    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = getattr(record, "context_str", "")
        return f"{base}  [{context}]" if context else base


def set_log_context(**fields: Any) -> None:
    """Set context for the current thread (replaces any previous context)."""
    _context.data = {k: v for k, v in fields.items() if v is not None}


def clear_log_context() -> None:
    _context.data = {}


class log_context:  # noqa: N801 - used as a lowercase context manager
    """Scoped log context: ``with log_context(job_id=..., clip_id=...):``."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields
        self._previous: dict[str, Any] = {}

    def __enter__(self) -> log_context:
        self._previous = dict(getattr(_context, "data", None) or {})
        merged = dict(self._previous)
        merged.update({k: v for k, v in self._fields.items() if v is not None})
        _context.data = merged
        return self

    def __exit__(self, *exc: object) -> None:
        _context.data = self._previous


def configure_logging(
    settings: Settings,
    *,
    console: bool = True,
    force: bool = False,
    log_filename: str = LOG_FILENAME,
) -> Path:
    """Install handlers. Idempotent unless ``force`` is set.

    Returns:
        The path of the rotating log file.
    """
    global _configured
    with _lock:
        log_dir = settings.paths.require("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_filename

        if _configured and not force:
            return log_path

        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

        root.setLevel(logging.DEBUG)
        context_filter = ContextFilter()

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=settings.logging.max_bytes,
            backupCount=settings.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(_level(settings.logging.file_level))
        file_handler.setFormatter(
            JsonFormatter()
            if settings.logging.json_logs
            else logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s %(context_str)s"
            )
        )
        file_handler.addFilter(context_filter)
        root.addHandler(file_handler)

        if console:
            stream = logging.StreamHandler(sys.stderr)
            stream.setLevel(_level(settings.logging.level))
            stream.setFormatter(ConsoleFormatter())
            stream.addFilter(context_filter)
            root.addHandler(stream)

        # Third-party chatter that adds nothing to an overnight log.
        for noisy in ("httpx", "httpcore", "urllib3", "PIL", "matplotlib", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.engine").setLevel(
            logging.INFO if settings.database.echo else logging.WARNING
        )

        _configured = True
        logging.getLogger(APP_SLUG).debug("logging configured -> %s", log_path)
        return log_path


def _level(name: str) -> int:
    resolved = logging.getLevelName(name.upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Package-scoped logger factory."""
    return logging.getLogger(APP_SLUG if name is None else f"{APP_SLUG}.{name}")
