"""Engine and session management.

The pipeline runs many worker threads against one database. On SQLite that
needs care, and all of it is concentrated here:

* **WAL** so the dashboard can read while workers write.
* **busy_timeout** so a contended write waits instead of raising.
* **BEGIN IMMEDIATE** for write transactions, which takes the write lock up
  front and eliminates the "upgrade deadlock" that otherwise bites when two
  threads read-then-write the same rows.
* **Foreign keys ON**, because SQLite disables them by default and the whole
  cascade design depends on them.

Every one of those is a no-op on PostgreSQL, so the same code path serves both.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from ..config import Settings
from ..errors import DatabaseError

log = logging.getLogger(__name__)


class Database:
    """Owns one engine and hands out sessions.

    Thread-safe: sessions are created per call, never shared across threads.
    """

    def __init__(self, settings: Settings, *, url: Optional[str] = None) -> None:
        self.settings = settings
        self.url = url or settings.database_url
        self.is_sqlite = self.url.startswith("sqlite")
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker[Session]] = None
        self._lock = threading.Lock()

    # -- engine -------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = self._create_engine()
        return self._engine

    def _create_engine(self) -> Engine:
        cfg = self.settings.database
        kwargs: dict[str, Any] = {"echo": cfg.echo, "future": True}

        if self.is_sqlite:
            self._ensure_sqlite_dir()
            # SQLite connections are cheap; a shared pool across threads adds
            # lock contention without adding throughput.
            kwargs["poolclass"] = NullPool
            kwargs["connect_args"] = {
                "check_same_thread": False,
                "timeout": cfg.sqlite_busy_timeout_ms / 1000.0,
            }
        else:
            kwargs["poolclass"] = QueuePool
            kwargs["pool_size"] = cfg.pool_size
            kwargs["max_overflow"] = cfg.max_overflow
            kwargs["pool_pre_ping"] = True

        engine = create_engine(self.url, **kwargs)
        if self.is_sqlite:
            _install_sqlite_pragmas(engine, cfg.sqlite_wal, cfg.sqlite_busy_timeout_ms,
                                    cfg.sqlite_synchronous, cfg.sqlite_cache_size_kb)
        log.debug("database engine ready: %s", _redact(self.url))
        return engine

    def _ensure_sqlite_dir(self) -> None:
        prefix = "sqlite:///"
        if not self.url.startswith(prefix):
            return
        db_path = Path(self.url[len(prefix) :])
        if str(db_path) not in {":memory:", ""}:
            db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def session_factory(self) -> sessionmaker[Session]:
        if self._session_factory is None:
            self._session_factory = sessionmaker(
                bind=self.engine, expire_on_commit=False, future=True
            )
        return self._session_factory

    # -- sessions -----------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Read-oriented session. Commits on success, rolls back on error."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def write_session(self) -> Iterator[Session]:
        """Write session that takes SQLite's write lock immediately.

        Use this for any transaction that reads a row and then updates it --
        claiming a job, resolving duplicates, incrementing counters. Without it
        two threads can both hold read locks and deadlock on upgrade.
        """
        session = self.session_factory()
        try:
            if self.is_sqlite:
                session.execute(text("BEGIN IMMEDIATE"))
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -- lifecycle ----------------------------------------------------------

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None

    def healthcheck(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as exc:
            log.error("database healthcheck failed: %s", exc)
            return False
        return True

    def vacuum(self) -> None:
        """Reclaim space and refresh planner statistics. Run after big purges."""
        statements = ["VACUUM", "ANALYZE"] if self.is_sqlite else ["VACUUM ANALYZE"]
        with self.engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            for statement in statements:
                conn.execute(text(statement))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Database {_redact(self.url)}>"


def _install_sqlite_pragmas(
    engine: Engine, wal: bool, busy_timeout_ms: int, synchronous: str, cache_kb: int
) -> None:
    """Apply per-connection PRAGMAs. SQLite forgets them on every connect."""

    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            if wal:
                cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA synchronous={synchronous}")
            # Negative cache_size is measured in KiB rather than pages.
            cursor.execute(f"PRAGMA cache_size=-{int(cache_kb)}")
            cursor.execute("PRAGMA temp_store=MEMORY")
            # 256 MiB mmap: large win when scanning the clips table for dedupe.
            cursor.execute("PRAGMA mmap_size=268435456")
        finally:
            cursor.close()


def _redact(url: str) -> str:
    """Hide credentials in a database URL before it reaches a log."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


_default: Optional[Database] = None
_default_lock = threading.Lock()


def get_database(settings: Settings) -> Database:
    """Process-wide database handle keyed to the settings' URL."""
    global _default
    with _default_lock:
        if _default is None or _default.url != settings.database_url:
            if _default is not None:
                _default.dispose()
            _default = Database(settings)
        return _default


def reset_database() -> None:
    """Drop the process-wide handle (tests, config reloads)."""
    global _default
    with _default_lock:
        if _default is not None:
            _default.dispose()
        _default = None


def require_session(session: Optional[Session]) -> Session:
    if session is None:
        raise DatabaseError("a database session is required but was not provided")
    return session
