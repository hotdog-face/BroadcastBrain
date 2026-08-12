"""The local dashboard.

A small FastAPI application serving a single static page plus a JSON API. It is
deliberately read-mostly: the one mutating endpoint re-queues analysis for a
clip, because "this one was classified wrong, run it again" is the thing you
actually want while looking at results.

It binds to localhost by default and serves media only from inside the
configured library and processing directories -- an unauthenticated local UI
must not become a way to read arbitrary files off the machine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..errors import BroadcastBrainError

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings) -> Any:
    """Build the FastAPI application.

    Imported lazily so that ``import broadcastbrain`` does not require FastAPI.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover - guarded by the CLI
        raise BroadcastBrainError(
            "the dashboard needs: pip install 'broadcastbrain[dashboard]'"
        ) from exc

    from sqlalchemy import select

    from .. import __version__, capabilities
    from ..db.base import get_database
    from ..db.models import Clip, Event
    from ..db.queue import JobQueue
    from ..db.repo import (
        AnalysisRepository,
        ClipRepository,
        DuplicateRepository,
        EntityRepository,
        EventRepository,
        TagRepository,
        TextRepository,
    )
    from ..pipeline.reports import latest_report
    from ..search.service import SearchService
    from ..types import JobKind
    from ..util.atomic import directory_size_bytes, free_space_gb

    db = get_database(settings)
    queue = JobQueue(db)
    search_service = SearchService(settings, db)

    app = FastAPI(title="BroadcastBrain", version=__version__, docs_url="/api/docs")

    # -- helpers -----------------------------------------------------------

    def _safe_media_path(raw: Optional[str]) -> Optional[Path]:
        """Resolve a stored media path, refusing anything outside our roots."""
        if not raw:
            return None
        candidate = Path(raw).resolve()
        roots = [
            settings.paths.require("library").resolve(),
            settings.paths.require("processing").resolve(),
            settings.paths.require("duplicates").resolve(),
            settings.paths.require("thumbnails").resolve(),
        ]
        for root in roots:
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            return candidate if candidate.exists() else None
        return None

    def _clip_payload(clip: Clip) -> dict[str, Any]:
        return {
            "id": clip.id,
            "title": clip.title,
            "category": clip.category,
            "subcategory": clip.subcategory,
            "confidence": round(clip.category_confidence, 3),
            "network": clip.network,
            "series": clip.series,
            "air_year": clip.air_year,
            "duration_s": round(clip.duration_s, 2),
            "summary": clip.summary,
            "status": clip.status,
            "stage": clip.stage,
            "is_alternate": clip.is_alternate,
            "quality_score": round(clip.quality_score, 3),
            "width": clip.width,
            "height": clip.height,
            "has_thumbnail": bool(clip.thumbnail_path),
            "has_media": bool(clip.file_path or clip.processing_path),
        }

    # -- API ---------------------------------------------------------------

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        """Queue, library and duplicate counts."""
        with db.session() as session:
            clip_stats = ClipRepository.statistics(session)
            duplicate_stats = DuplicateRepository.statistics(session)
        return {
            "version": __version__,
            "instance": settings.instance_name,
            "queue": queue.counts_by_state(),
            "queue_by_kind": queue.counts_by_kind(),
            "clips": clip_stats,
            "duplicates": duplicate_stats,
            "era": list(settings.era_range()),
        }

    @app.get("/api/system")
    def api_system() -> dict[str, Any]:
        """Live CPU, memory, GPU and storage usage."""
        cpu_percent: Optional[float] = None
        memory_percent: Optional[float] = None
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
        except ImportError:
            pass

        library = settings.paths.require("library")
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "gpu": capabilities.gpu_snapshot().as_dict(),
            "storage": {
                "library_bytes": directory_size_bytes(library),
                "free_gb": round(free_space_gb(library), 1),
                "path": str(library),
            },
        }

    @app.get("/api/queue")
    def api_queue(limit: int = Query(25, le=200)) -> dict[str, Any]:
        """Currently running jobs and recent failures."""
        return {
            "counts": queue.counts_by_state(),
            "running": [
                {
                    "id": job.id,
                    "kind": job.kind,
                    "worker": job.worker_id,
                    "progress": round(job.progress, 3),
                    "note": job.progress_note,
                    "attempts": job.attempts,
                    "started_at": job.started_at.isoformat() if job.started_at else None,
                }
                for job in queue.running_jobs(limit=limit)
            ],
            "failures": [
                {
                    "id": job.id,
                    "kind": job.kind,
                    "state": job.state,
                    "attempts": job.attempts,
                    "error": job.last_error,
                }
                for job in queue.recent_failures(limit=limit)
            ],
        }

    @app.get("/api/clips")
    def api_clips(
        limit: int = Query(30, le=200),
        category: Optional[str] = None,
        include_alternates: bool = False,
    ) -> dict[str, Any]:
        """Most recently completed clips."""
        statement = select(Clip)
        if not include_alternates:
            statement = statement.where(Clip.is_alternate.is_(False))
        if category:
            statement = statement.where(Clip.category == category)
        with db.session() as session:
            rows = list(
                session.execute(
                    statement.order_by(Clip.updated_at.desc()).limit(limit)
                ).scalars()
            )
            return {"clips": [_clip_payload(clip) for clip in rows]}

    @app.get("/api/clip/{clip_id}")
    def api_clip(clip_id: str) -> dict[str, Any]:
        """Everything about one clip, including its classification evidence."""
        with db.session() as session:
            clip = ClipRepository.get(session, clip_id)
            if clip is None:
                raise HTTPException(status_code=404, detail="clip not found")
            payload = _clip_payload(clip)
            payload["tags"] = TagRepository.for_clip(session, clip_id)
            payload["entities"] = [
                {"name": entity.name, "kind": entity.kind, "confidence": round(confidence, 3)}
                for entity, confidence in EntityRepository.for_clip(session, clip_id)
            ]
            texts = TextRepository.all_for_clip(session, clip_id)
            payload["transcript"] = texts.get("transcript", "")
            payload["ocr"] = texts.get("ocr", "")
            payload["evidence"] = clip.category_evidence or []
            payload["analyses"] = [
                {
                    "analyzer": name,
                    "status": record.status,
                    "version": record.analyzer_version,
                    "backend": record.backend,
                    "error": record.error,
                    "duration_ms": record.duration_ms,
                }
                for name, record in sorted(AnalysisRepository.for_clip(session, clip_id).items())
            ]
        return payload

    @app.post("/api/clip/{clip_id}/reanalyze")
    def api_reanalyze(clip_id: str, analyzer: Optional[str] = None) -> dict[str, Any]:
        """Queue a clip for re-analysis (all analyzers, or just one)."""
        with db.session() as session:
            if ClipRepository.get(session, clip_id) is None:
                raise HTTPException(status_code=404, detail="clip not found")
        job_id = queue.enqueue(
            JobKind.ANALYZE_CLIP,
            {"clip_id": clip_id, "force": [analyzer] if analyzer else ["*"]},
            requeue_if_done=True,
        )
        return {"queued": job_id is not None, "job_id": job_id, "clip_id": clip_id}

    @app.get("/api/search")
    def api_search(
        q: str = Query(..., min_length=1),
        limit: int = Query(30, le=200),
        include_alternates: bool = False,
    ) -> dict[str, Any]:
        """Natural-language search over the library."""
        response = search_service.search(q, limit=limit, include_alternates=include_alternates)
        return response.as_dict()

    @app.get("/api/similar/{clip_id}")
    def api_similar(clip_id: str, limit: int = Query(20, le=100)) -> dict[str, Any]:
        return search_service.similar(clip_id, limit=limit).as_dict()

    @app.get("/api/facets")
    def api_facets() -> dict[str, Any]:
        return search_service.facets()

    @app.get("/api/duplicates")
    def api_duplicates(limit: int = Query(25, le=100)) -> dict[str, Any]:
        with db.session() as session:
            groups = DuplicateRepository.recent(session, limit=limit)
            payload = []
            for group in groups:
                canonical = ClipRepository.get(session, group.canonical_clip_id)
                payload.append(
                    {
                        "id": group.id,
                        "method": group.method,
                        "members": group.member_count,
                        "canonical": _clip_payload(canonical) if canonical else None,
                    }
                )
            statistics = DuplicateRepository.statistics(session)
        return {"groups": payload, "statistics": statistics}

    @app.get("/api/events")
    def api_events(limit: int = Query(50, le=200), level: Optional[str] = None) -> dict[str, Any]:
        with db.session() as session:
            events: list[Event] = EventRepository.recent(session, limit=limit, level=level)
            return {
                "events": [
                    {
                        "at": event.created_at.isoformat(),
                        "level": event.level,
                        "kind": event.kind,
                        "message": event.message,
                        "clip_id": event.clip_id,
                    }
                    for event in events
                ]
            }

    @app.get("/api/report")
    def api_report() -> dict[str, Any]:
        return latest_report(settings) or {"detail": "no report yet"}

    @app.get("/api/capabilities")
    def api_capabilities() -> dict[str, Any]:
        from ..analyze import all_analyzers

        return {
            "backends": capabilities.summary(),
            "analyzers": [a.describe() for a in all_analyzers()],
        }

    # -- media -------------------------------------------------------------

    @app.get("/thumb/{clip_id}")
    def thumbnail(clip_id: str) -> Any:
        with db.session() as session:
            clip = ClipRepository.get(session, clip_id)
            raw = clip.thumbnail_path if clip else None
        path = _safe_media_path(raw)
        if path is None:
            raise HTTPException(status_code=404, detail="no thumbnail")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/media/{clip_id}")
    def media(clip_id: str) -> Any:
        """Serve a clip for in-browser preview."""
        if not settings.dashboard.allow_preview:
            raise HTTPException(status_code=403, detail="preview disabled in configuration")
        with db.session() as session:
            clip = ClipRepository.get(session, clip_id)
            raw = (clip.file_path or clip.processing_path) if clip else None
        path = _safe_media_path(raw)
        if path is None:
            raise HTTPException(status_code=404, detail="no media file")
        return FileResponse(path)

    # -- static ------------------------------------------------------------

    @app.get("/")
    def index() -> Any:
        page = STATIC_DIR / "index.html"
        if not page.exists():  # pragma: no cover - packaging problem
            return JSONResponse({"detail": "dashboard assets are missing"}, status_code=500)
        return FileResponse(page, media_type="text/html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def serve(settings: Settings, *, host: str, port: int) -> None:
    """Run the dashboard with uvicorn."""
    import uvicorn

    application = create_app(settings)
    uvicorn.run(application, host=host, port=port, log_level=settings.logging.level.lower())
