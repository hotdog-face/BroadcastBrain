"""The morning-after report.

Written to ``Logs/reports/`` as both Markdown and JSON at the end of every run.
Its job is to answer, in thirty seconds and without opening a database, the
questions you actually have after an overnight pass: what came in, what it
turned into, what went wrong, and what would improve the result next time.

The "recommendations" section is the part worth keeping honest -- it reports
capability gaps (no Whisper installed, no OCR) and quality signals (many
low-confidence clips, sources that segmented poorly) rather than congratulating
itself on the clips that worked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ..capabilities import summary as capability_summary
from ..config import Settings
from ..db.base import Database
from ..db.models import Clip, Source
from ..db.queue import JobQueue
from ..db.repo import (
    AnalysisRepository,
    ClipRepository,
    DuplicateRepository,
    EventRepository,
    SourceRepository,
)
from ..types import Category, SourceStatus
from ..util.atomic import directory_size_bytes, free_space_gb, write_json
from ..util.text import format_duration

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .orchestrator import RunSummary

log = logging.getLogger(__name__)

#: Clips below this classification confidence are worth a human glance.
LOW_CONFIDENCE_THRESHOLD = 0.4


@dataclass
class RunReport:
    """Everything the report needs, gathered once."""

    generated_at: str
    run: dict[str, Any]
    queue: dict[str, int]
    sources: dict[str, Any]
    clips: dict[str, Any]
    duplicates: dict[str, int]
    analyzers: dict[str, dict[str, int]]
    capabilities: list[dict[str, Any]]
    storage: dict[str, Any]
    failures: list[dict[str, Any]]
    recommendations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "run": self.run,
            "queue": self.queue,
            "sources": self.sources,
            "clips": self.clips,
            "duplicates": self.duplicates,
            "analyzers": self.analyzers,
            "capabilities": self.capabilities,
            "storage": self.storage,
            "failures": self.failures,
            "recommendations": self.recommendations,
        }


def collect_report(
    settings: Settings, db: Database, queue: JobQueue, summary: RunSummary
) -> RunReport:
    """Gather everything the report shows."""
    with db.session() as session:
        clip_stats = ClipRepository.statistics(session)
        source_counts = SourceRepository.counts_by_status(session)
        source_bytes = SourceRepository.total_bytes(session)
        duplicate_stats = DuplicateRepository.statistics(session)
        analyzer_stats = AnalysisRepository.counts_by_status(session)
        low_confidence = _low_confidence_clips(session)
        poor_sources = _poorly_segmented(session)
        recent_failures = [
            {
                "kind": event.kind,
                "message": event.message,
                "at": event.created_at.isoformat(),
                "level": event.level,
            }
            for event in EventRepository.recent(session, limit=25, level="error")
        ]

    library_root = settings.paths.require("library")
    storage = {
        "library_bytes": directory_size_bytes(library_root),
        "source_bytes": source_bytes,
        "free_gb": round(free_space_gb(library_root), 1),
        "library_path": str(library_root),
    }

    report = RunReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        run=summary.as_dict(),
        queue=queue.counts_by_state(),
        sources={"by_status": source_counts, "poorly_segmented": poor_sources},
        clips={**clip_stats, "low_confidence": low_confidence},
        duplicates=duplicate_stats,
        analyzers=analyzer_stats,
        capabilities=capability_summary(),
        storage=storage,
        failures=recent_failures,
    )
    report.recommendations = _recommendations(report, settings)
    return report


def write_report(settings: Settings, report: RunReport) -> Path:
    """Write ``report-<timestamp>.md`` and ``.json``; also refresh ``latest``."""
    directory = settings.paths.require("logs") / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = directory / f"report-{stamp}.json"
    write_json(json_path, report.as_dict())
    write_json(directory / "latest.json", report.as_dict())

    markdown = render_markdown(report)
    markdown_path = directory / f"report-{stamp}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    (directory / "latest.md").write_text(markdown, encoding="utf-8")

    log.info("report written to %s", markdown_path)
    return markdown_path


def render_markdown(report: RunReport) -> str:
    """Render the report as Markdown."""
    run = report.run
    clips = report.clips
    lines: list[str] = [
        "# BroadcastBrain run report",
        "",
        f"Generated {report.generated_at}",
        "",
        "## What happened",
        "",
        f"- Duration: **{format_duration(float(run.get('duration_s') or 0))}**",
        f"- Clips created this run: **{run.get('clips_created', 0)}**",
        f"- Jobs completed: **{(run.get('workers') or {}).get('completed', 0)}**, "
        f"failed: {(run.get('workers') or {}).get('dead', 0)}",
        f"- Recovered from a previous run: {run.get('recovered_jobs', 0)} job(s), "
        f"{run.get('requeued_items', 0)} item(s)",
    ]

    discovery = run.get("discovery")
    if discovery:
        lines.append(
            f"- Discovery: {discovery.get('new_sources', 0)} new, "
            f"{discovery.get('already_known', 0)} already known, "
            f"{discovery.get('rejected', 0)} rejected"
        )

    lines += [
        "",
        "## Library",
        "",
        f"- Total clips: **{clips.get('total_clips', 0)}** "
        f"({clips.get('unique_clips', 0)} unique, {clips.get('alternates', 0)} alternates)",
        f"- Total runtime: **{format_duration(float(clips.get('total_duration_s') or 0))}**",
        f"- Unknown (kept, not discarded): {clips.get('unknown_clips', 0)}",
        f"- Duplicate groups: {report.duplicates.get('duplicate_groups', 0)}",
        "",
        "### By category",
        "",
        "| Category | Clips |",
        "| --- | ---: |",
    ]
    by_category = clips.get("by_category") or {}
    for category in Category:
        count = by_category.get(category.value, 0)
        if count:
            lines.append(f"| {category.value} | {count} |")

    lines += ["", "## Analyzers", "", "| Analyzer | ok | skipped | failed |", "| --- | ---: | ---: | ---: |"]
    for name, statuses in sorted(report.analyzers.items()):
        lines.append(
            f"| {name} | {statuses.get('ok', 0)} | {statuses.get('skipped', 0)} | "
            f"{statuses.get('failed', 0)} |"
        )

    missing = [c for c in report.capabilities if not c.get("available")]
    if missing:
        lines += ["", "## Missing capabilities", ""]
        for capability in missing:
            hint = capability.get("install_hint") or ""
            lines.append(f"- **{capability['name']}** -- {hint}")

    storage = report.storage
    lines += [
        "",
        "## Storage",
        "",
        f"- Library: {storage.get('library_bytes', 0) / (1024**3):.2f} GB at `{storage.get('library_path')}`",
        f"- Source material ingested: {storage.get('source_bytes', 0) / (1024**3):.2f} GB",
        f"- Free space: {storage.get('free_gb', 0)} GB",
    ]

    if report.failures:
        lines += ["", "## Errors", ""]
        for failure in report.failures[:15]:
            lines.append(f"- `{failure['at']}` **{failure['kind']}**: {failure['message']}")

    if report.recommendations:
        lines += ["", "## What would improve the next run", ""]
        lines.extend(f"- {item}" for item in report.recommendations)

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis of the results
# ---------------------------------------------------------------------------


def _low_confidence_clips(session: Any, *, limit: int = 25) -> dict[str, Any]:
    from sqlalchemy import func, select

    total = int(
        session.execute(
            select(func.count())
            .select_from(Clip)
            .where(Clip.category_confidence < LOW_CONFIDENCE_THRESHOLD)
        ).scalar_one()
    )
    examples = [
        {
            "id": clip.id,
            "title": clip.title,
            "category": clip.category,
            "confidence": round(clip.category_confidence, 3),
            "duration_s": round(clip.duration_s, 1),
        }
        for clip in session.execute(
            select(Clip)
            .where(Clip.category_confidence < LOW_CONFIDENCE_THRESHOLD)
            .order_by(Clip.category_confidence.asc())
            .limit(limit)
        ).scalars()
    ]
    return {"count": total, "threshold": LOW_CONFIDENCE_THRESHOLD, "examples": examples}


def _poorly_segmented(session: Any, *, limit: int = 15) -> list[dict[str, Any]]:
    """Sources whose segmentation looks suspicious.

    One clip out of a three-hour recording almost always means the detectors
    found nothing usable -- worth re-running with different thresholds rather
    than silently accepting a three-hour "clip".
    """
    from sqlalchemy import select

    suspicious: list[dict[str, Any]] = []
    for source in session.execute(
        select(Source)
        .where(Source.status == SourceStatus.COMPLETE.value, Source.duration_s.is_not(None))
        .order_by(Source.completed_at.desc())
        .limit(500)
    ).scalars():
        duration = source.duration_s or 0.0
        if duration < 1800:
            continue
        clips_per_hour = (source.clip_count or 0) / max(duration / 3600.0, 1e-6)
        if clips_per_hour < 2.0:
            suspicious.append(
                {
                    "id": source.id,
                    "filename": source.filename,
                    "duration_s": round(duration, 1),
                    "clip_count": source.clip_count,
                    "clips_per_hour": round(clips_per_hour, 2),
                }
            )
        if len(suspicious) >= limit:
            break
    return suspicious


def _recommendations(report: RunReport, settings: Settings) -> list[str]:
    """Concrete, actionable next steps -- not praise."""
    tips: list[str] = []

    missing = {c["name"]: c for c in report.capabilities if not c.get("available")}
    if "ffmpeg" in missing:
        tips.append(
            "**ffmpeg is not installed.** Nothing can be segmented or cut without it: "
            + str(missing["ffmpeg"].get("install_hint", ""))
        )
    if "whisper" in missing:
        tips.append(
            "Install Whisper (`pip install 'broadcastbrain[speech]'`) then run "
            "`bb reanalyze --missing` -- transcripts are the strongest search signal."
        )
    if "pytesseract" in missing or "tesseract" in missing:
        tips.append(
            "Install OCR (`pip install 'broadcastbrain[ocr]'` plus the tesseract binary) "
            "then `bb reanalyze --missing`. On-screen text is what dates and attributes clips."
        )
    if "sentence_transformers" in missing:
        tips.append(
            "Install embeddings (`pip install 'broadcastbrain[embeddings]'`) to enable "
            "semantic search and 'find clips like this one'."
        )
    if "chromaprint" in missing:
        tips.append(
            "Install chromaprint (`brew install chromaprint`) to catch duplicates that "
            "differ visually but share audio."
        )

    low = report.clips.get("low_confidence") or {}
    if int(low.get("count", 0)) > 0:
        total = int(report.clips.get("total_clips") or 1)
        share = int(low["count"]) / max(1, total)
        if share > 0.25:
            tips.append(
                f"{low['count']} clip(s) ({share:.0%}) classified below "
                f"{LOW_CONFIDENCE_THRESHOLD:.0%} confidence. Installing the missing "
                "analyzers above will resolve most of them without re-cutting anything."
            )

    poor = report.sources.get("poorly_segmented") or []
    if poor:
        tips.append(
            f"{len(poor)} long source(s) produced very few clips -- segmentation likely "
            "found no boundaries. Try `bb resegment <source-id>` with a lower "
            "`segmentation.min_boundary_score`."
        )

    if int(report.queue.get("dead", 0)) > 0:
        tips.append(
            f"{report.queue['dead']} job(s) exhausted their retries. Inspect with "
            "`bb jobs --failed`, then `bb retry` once the cause is fixed."
        )

    free_gb = float(report.storage.get("free_gb") or 0)
    if free_gb < settings.workers.min_free_gb * 2:
        tips.append(
            f"Only {free_gb:.0f} GB free on the library volume. The pipeline pauses "
            f"below {settings.workers.min_free_gb} GB."
        )

    unknown = int(report.clips.get("unknown_clips", 0))
    total_clips = int(report.clips.get("total_clips", 0))
    if total_clips and unknown / total_clips > 0.3:
        tips.append(
            f"{unknown} of {total_clips} clips are Unknown. They are kept and searchable; "
            "adding networks or brands to `analyze/knowledge.py` and running "
            "`bb recategorize` will reclassify them in seconds."
        )
    return tips


def latest_report(settings: Settings) -> Optional[dict[str, Any]]:
    """Load the most recent report, for the dashboard."""
    path = settings.paths.require("logs") / "reports" / "latest.json"
    if not path.exists():
        return None
    from ..util.atomic import read_json

    try:
        data = read_json(path)
    except (OSError, ValueError):  # pragma: no cover - corrupted report
        return None
    return data if isinstance(data, dict) else None
