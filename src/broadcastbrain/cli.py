"""``bb`` -- the command line.

One command does the job the brief describes::

    bb run --watch

Everything else exists to inspect, correct or extend what that command
produced. Commands are grouped by what you would be doing at the time:

* **Setup**: ``init``, ``doctor``, ``config``
* **Processing**: ``run``, ``watch``, ``scan``, ``export``
* **Inspection**: ``status``, ``clips``, ``clip``, ``jobs``, ``report``
* **Correction**: ``reanalyze``, ``recategorize``, ``resegment``, ``retry``
* **Discovery**: ``search``, ``similar``, ``fetch``
* **UI**: ``dashboard``
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__, capabilities
from .config import Settings, load_settings
from .constants import APP_NAME, DEFAULT_CONFIG_FILENAME
from .errors import BroadcastBrainError
from .logging_setup import configure_logging
from .util.text import format_duration

console = Console()
error_console = Console(stderr=True, style="bold red")
log = logging.getLogger(__name__)

app = typer.Typer(
    name="bb",
    help=f"{APP_NAME} -- autonomous television ingestion for Fieldstation 42.",
    no_args_is_help=True,
    add_completion=True,
)

#: Populated by the root callback so every command shares one Settings.
_state: dict[str, Any] = {}


def _settings() -> Settings:
    settings = _state.get("settings")
    if settings is None:
        settings = load_settings()
        _state["settings"] = settings
    return settings


def _pipeline() -> Any:
    from .pipeline.orchestrator import Pipeline

    pipeline = _state.get("pipeline")
    if pipeline is None:
        pipeline = Pipeline(_settings())
        pipeline.initialize()
        _state["pipeline"] = pipeline
    return pipeline


@app.callback()
def main_callback(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to a TOML config file."
    ),
    root: Optional[Path] = typer.Option(
        None, "--root", "-r", help="Data root directory (overrides paths.root)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan everything, write nothing."),
) -> None:
    """Global options, applied before any command runs."""
    overrides: dict[str, Any] = {}
    if root is not None:
        overrides["paths"] = {"root": str(root)}
    if dry_run:
        overrides["dry_run"] = True

    try:
        settings = load_settings(config, **overrides)
    except BroadcastBrainError as exc:
        error_console.print(f"configuration error: {exc}")
        raise typer.Exit(code=2) from exc

    if verbose:
        settings.logging.level = "DEBUG"
    elif quiet:
        settings.logging.level = "WARNING"

    settings.paths.ensure()
    configure_logging(settings)
    _state["settings"] = settings


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


@app.command()
def init(
    write_config: bool = typer.Option(
        True, "--write-config/--no-write-config", help="Write a starter TOML config."
    ),
) -> None:
    """Create the directory tree and database. Safe to re-run."""
    settings = _settings()
    settings.paths.ensure()

    from .library.layout import ensure_library_tree

    ensure_library_tree(settings.paths.require("library"))

    pipeline = _pipeline()
    version = pipeline.initialize()

    console.print(f"[green]{APP_NAME} initialized[/green]")
    console.print(f"  data root:  {settings.paths.root}")
    console.print(f"  database:   {settings.database_url}")
    console.print(f"  schema:     v{version}")

    if write_config:
        target = Path.cwd() / DEFAULT_CONFIG_FILENAME
        if target.exists():
            console.print(f"  config:     {target} (left alone; already exists)")
        else:
            target.write_text(_starter_config(settings), encoding="utf-8")
            console.print(f"  config:     {target} (written)")

    console.print(
        f"\nDrop recordings into [bold]{settings.paths.require('incoming')}[/bold] "
        "and run [bold]bb run[/bold]."
    )


@app.command()
def doctor() -> None:
    """Report what is installed and what each missing piece would buy you."""
    settings = _settings()
    from .media.ffmpeg import FFmpeg

    table = Table(title="Capabilities", show_lines=False)
    table.add_column("Backend", style="cyan")
    table.add_column("Status")
    table.add_column("Detail / how to install", overflow="fold")

    for capability in capabilities.detect_all(
        settings.ffmpeg_binary, settings.ffprobe_binary
    ).values():
        status = "[green]yes[/green]" if capability.available else "[yellow]no[/yellow]"
        detail = capability.detail if capability.available else capability.install_hint
        # Escaped: install hints contain `[vision]`-style extras, which Rich
        # would otherwise swallow as markup and print as `pip install
        # 'broadcastbrain'` -- advice that silently does the wrong thing.
        table.add_row(capability.name, status, escape(detail or ""))
    console.print(table)

    gpu = capabilities.detect_gpu()
    console.print(f"\nGPU: {gpu.vendor} {gpu.name}" if gpu.available else "\nGPU: none detected")
    console.print(f"ffmpeg: {FFmpeg(settings).version()}")

    from .analyze import all_analyzers

    unavailable = [a for a in all_analyzers() if not a.available()]
    if unavailable:
        console.print(
            f"\n[yellow]{len(unavailable)} analyzer(s) will be skipped[/yellow] "
            "and recorded as such; install the backends above and run "
            "[bold]bb reanalyze --missing[/bold] to backfill without redoing anything else."
        )
    else:
        console.print("\n[green]Every analyzer has its backend installed.[/green]")

    if not capabilities.has("ffmpeg"):
        error_console.print("\nffmpeg is required. Nothing can be processed without it.")
        raise typer.Exit(code=1)


@app.command("config")
def show_config(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output.")
) -> None:
    """Show the effective configuration and where it came from."""
    settings = _settings()
    if as_json:
        console.print_json(json.dumps(settings.model_dump(mode="json"), default=str))
        return

    console.print(f"[bold]{APP_NAME}[/bold] {__version__}")
    console.print(f"config file: {settings.config_file or '(none; defaults + environment)'}")
    for key, value in settings.describe().items():
        console.print(f"  {key:<16} {value}")

    table = Table(title="Paths")
    table.add_column("Name", style="cyan")
    table.add_column("Location", overflow="fold")
    from .config import DEFAULT_DIRECTORY_NAMES

    table.add_row("root", str(settings.paths.root))
    for name in DEFAULT_DIRECTORY_NAMES:
        table.add_row(name, str(settings.paths.require(name)))
    console.print(table)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


@app.command()
def run(
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Keep running and process new files as they appear."
    ),
    no_scan: bool = typer.Option(False, "--no-scan", help="Skip the Incoming scan."),
    no_export: bool = typer.Option(False, "--no-export", help="Skip the Fieldstation export."),
    poll: float = typer.Option(30.0, "--poll", help="Watch interval in seconds."),
) -> None:
    """Process everything in Incoming. This is the overnight command."""
    pipeline = _pipeline()
    settings = _settings()

    if settings.dry_run:
        console.print("[yellow]dry run: nothing will be written[/yellow]")

    if watch:
        from .pipeline.watcher import Watcher

        watcher = Watcher(pipeline, poll_seconds=poll)
        watcher.install_signal_handlers()
        console.print(
            f"[green]watching[/green] {settings.paths.require('incoming')} "
            f"(every {poll:.0f}s). Ctrl-C to stop."
        )
        state = watcher.run()
        console.print(f"stopped after {state.scans} scan(s); {state.sources_found} source(s) found")
        return

    with console.status("[bold]processing[/bold]", spinner="dots") as status:

        def on_tick(snapshot: dict[str, Any]) -> None:
            current = ", ".join(f"{k}: {v}" for k, v in (snapshot.get("current") or {}).items())
            status.update(
                f"[bold]processing[/bold] -- {snapshot.get('completed', 0)} done"
                + (f" | {current}" if current else "")
            )

        summary = pipeline.run(scan=not no_scan, export=not no_export, on_tick=on_tick)

    _print_run_summary(summary)


@app.command()
def watch(poll: float = typer.Option(30.0, "--poll", help="Seconds between scans.")) -> None:
    """Watch Incoming and process new files forever."""
    run(watch=True, no_scan=False, no_export=False, poll=poll)


@app.command()
def scan() -> None:
    """Scan Incoming and queue new files, without processing them."""
    pipeline = _pipeline()
    result = pipeline.scan_incoming()
    console.print(result.summary())


@app.command()
def export(
    limit: Optional[int] = typer.Option(None, "--limit", help="Cap the number of clips.")
) -> None:
    """Regenerate the Fieldstation export."""
    pipeline = _pipeline()
    from .export.fieldstation import FieldstationExporter

    summary = FieldstationExporter(_settings(), pipeline.db).export(limit=limit)
    console.print(f"[green]export written[/green] -> {summary.root}")
    for key, value in summary.as_dict().items():
        console.print(f"  {key:<16} {value}")


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


@app.command("fs42")
def fieldstation42(
    channel_start: int = typer.Option(3, "--channel-start", help="First channel number."),
    validate_only: bool = typer.Option(False, "--validate", help="Only check an existing export."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Build the export here (e.g. an external drive)."
    ),
    content_root: Optional[str] = typer.Option(
        None,
        "--content-root",
        help="Path to write into each config's content_dir, as the TARGET machine "
        "will see it. Use 'catalog' when the catalog folder will sit inside the "
        "FieldStation42 directory.",
    ),
) -> None:
    """Build a FieldStation42 catalog tree and station configs."""
    from .export.fieldstation42 import Fieldstation42Exporter, validate_export

    settings = _settings()
    pipeline = _pipeline()
    exporter = Fieldstation42Exporter(
        settings, pipeline.db, output_root=output, content_root=content_root
    )

    if not validate_only:
        result = exporter.export(first_channel=channel_start)
        table = Table(title="FieldStation42 channels")
        table.add_column("ch", justify="right")
        table.add_column("network", style="cyan")
        table.add_column("programming", justify="right")
        table.add_column("ads", justify="right")
        table.add_column("bumps", justify="right")
        table.add_column("ready")
        for station in result.stations:
            info = station.summary()
            table.add_row(
                str(info["channel_number"]), str(info["network"]),
                f"{info['programme_minutes']} min", str(info["commercials"]),
                str(info["bumps"]),
                "[green]yes[/green]" if info["schedulable"] else "[yellow]no[/yellow]",
            )
        console.print(table)
        console.print(
            f"\n{result.files_linked} file(s) hardlinked "
            f"({result.files_copied} copied) -> {result.root}"
        )
        for station in result.stations:
            for warning in station.warnings:
                console.print(f"  [yellow]{station.network}[/yellow]: {warning}")

    problems = validate_export(exporter.root)
    failed = {name: issues for name, issues in problems.items() if issues}
    if failed:
        error_console.print(f"\n{len(failed)} config(s) have problems:")
        for name, issues in failed.items():
            for issue in issues:
                error_console.print(f"  {name}: {issue}")
        raise typer.Exit(code=1)
    console.print(f"\n[green]{len(problems)} config(s) validated[/green]")
    console.print(f"Install: copy {exporter.root / 'confs'}/*.json into FieldStation42/confs/")


@app.command()
def status(as_json: bool = typer.Option(False, "--json")) -> None:
    """Show queue, library and duplicate statistics."""
    pipeline = _pipeline()
    snapshot = pipeline.status()
    if as_json:
        console.print_json(json.dumps(snapshot, default=str))
        return

    queue_table = Table(title="Queue")
    queue_table.add_column("State", style="cyan")
    queue_table.add_column("Jobs", justify="right")
    for state, count in snapshot["queue"].items():
        queue_table.add_row(state, str(count))
    console.print(queue_table)

    clips = snapshot["clips"]
    console.print(
        f"\n[bold]Library[/bold]: {clips['total_clips']} clips "
        f"({clips['unique_clips']} unique, {clips['alternates']} alternates), "
        f"{format_duration(clips['total_duration_s'])} of material"
    )

    category_table = Table(title="By category")
    category_table.add_column("Category", style="cyan")
    category_table.add_column("Clips", justify="right")
    for category, count in sorted(
        clips["by_category"].items(), key=lambda kv: kv[1], reverse=True
    ):
        category_table.add_row(category, str(count))
    console.print(category_table)

    console.print(
        f"\nSources: {snapshot['sources']}\nDuplicates: {snapshot['duplicates']}"
    )


@app.command()
def clips(
    category: Optional[str] = typer.Option(None, "--category", help="Filter by category."),
    network: Optional[str] = typer.Option(None, "--network", help="Filter by network."),
    year: Optional[int] = typer.Option(None, "--year", help="Filter by estimated year."),
    limit: int = typer.Option(25, "--limit", "-n"),
) -> None:
    """List clips in the library."""
    from sqlalchemy import select

    from .db.models import Clip

    pipeline = _pipeline()
    statement = select(Clip).where(Clip.is_alternate.is_(False))
    if category:
        statement = statement.where(Clip.category == category)
    if network:
        statement = statement.where(Clip.network == network)
    if year:
        statement = statement.where(Clip.air_year == year)

    with pipeline.db.session() as session:
        rows = list(
            session.execute(statement.order_by(Clip.created_at.desc()).limit(limit)).scalars()
        )

    table = Table(title=f"{len(rows)} clip(s)")
    table.add_column("id", style="dim")
    table.add_column("category", style="cyan")
    table.add_column("dur", justify="right")
    table.add_column("network")
    table.add_column("year", justify="right")
    table.add_column("title", overflow="fold")
    for clip in rows:
        table.add_row(
            clip.id[:8],
            f"{clip.category} ({clip.category_confidence:.2f})",
            format_duration(clip.duration_s),
            clip.network or "-",
            str(clip.air_year or "-"),
            clip.title or "-",
        )
    console.print(table)


@app.command()
def clip(clip_id: str = typer.Argument(..., help="Clip id or 8-character prefix.")) -> None:
    """Show everything known about one clip, including why it was classified."""
    from sqlalchemy import select

    from .db.models import Clip
    from .db.repo import AnalysisRepository, EntityRepository, TagRepository, TextRepository

    pipeline = _pipeline()
    with pipeline.db.session() as session:
        row = session.execute(
            select(Clip).where(Clip.id.like(f"{clip_id}%")).limit(1)
        ).scalar_one_or_none()
        if row is None:
            error_console.print(f"no clip matching {clip_id!r}")
            raise typer.Exit(code=1)
        tags = TagRepository.for_clip(session, row.id)
        entities = EntityRepository.for_clip(session, row.id)
        texts = TextRepository.all_for_clip(session, row.id)
        analyses = AnalysisRepository.for_clip(session, row.id)

    console.print(f"[bold]{row.title or 'Untitled'}[/bold]  [dim]{row.id}[/dim]")
    console.print(
        f"  {row.category} ({row.category_confidence:.2f})"
        + (f" / {row.subcategory}" if row.subcategory else "")
    )
    console.print(f"  {format_duration(row.duration_s)}  {row.width}x{row.height}  {row.video_codec}")
    console.print(f"  network: {row.network or '-'}  year: {row.air_year or '-'}  genre: {row.genre or '-'}")
    console.print(f"  file: {row.file_path or row.processing_path or '-'}")
    if row.summary:
        console.print(f"\n{row.summary}")
    if tags:
        console.print(f"\n[cyan]tags[/cyan]: {', '.join(tags[:30])}")
    if entities:
        listed = ", ".join(f"{e.name} ({e.kind}, {c:.2f})" for e, c in entities[:15])
        console.print(f"[cyan]entities[/cyan]: {listed}")

    if row.category_evidence:
        table = Table(title="Why it was classified this way")
        table.add_column("category", style="cyan")
        table.add_column("weight", justify="right")
        table.add_column("reason", overflow="fold")
        for item in row.category_evidence[:20]:
            table.add_row(
                str(item.get("category")), f"{float(item.get('weight', 0)):.2f}", str(item.get("reason"))
            )
        console.print(table)

    analysis_table = Table(title="Analyzers")
    analysis_table.add_column("analyzer", style="cyan")
    analysis_table.add_column("status")
    analysis_table.add_column("version")
    analysis_table.add_column("note", overflow="fold")
    for name, record in sorted(analyses.items()):
        analysis_table.add_row(
            name, record.status, record.analyzer_version, (record.error or record.backend or "")[:80]
        )
    console.print(analysis_table)

    transcript = texts.get("transcript")
    if transcript:
        console.print(f"\n[cyan]transcript[/cyan]: {transcript[:800]}")


@app.command()
def jobs(
    failed: bool = typer.Option(False, "--failed", help="Show only dead jobs."),
    limit: int = typer.Option(25, "--limit", "-n"),
) -> None:
    """Show the work queue."""
    pipeline = _pipeline()
    if failed:
        rows = pipeline.queue.recent_failures(limit=limit)
        title = "Dead jobs"
    else:
        rows = pipeline.queue.running_jobs(limit=limit)
        title = "Running jobs"

    table = Table(title=title)
    table.add_column("id", justify="right")
    table.add_column("kind", style="cyan")
    table.add_column("state")
    table.add_column("attempts", justify="right")
    table.add_column("detail", overflow="fold")
    for job in rows:
        table.add_row(
            str(job.id), job.kind, job.state, str(job.attempts),
            (job.last_error or job.progress_note or "")[:100],
        )
    console.print(table)
    console.print(f"\ncounts: {pipeline.queue.counts_by_state()}")


@app.command()
def report() -> None:
    """Show the most recent run report."""
    from .pipeline.reports import latest_report

    data = latest_report(_settings())
    if data is None:
        console.print("no report yet -- run [bold]bb run[/bold] first")
        return

    from .pipeline.reports import RunReport, render_markdown

    console.print(render_markdown(RunReport(**data)))


@app.command()
def analyzers() -> None:
    """List analyzers, their versions and whether their backends exist."""
    from .analyze import all_analyzers

    table = Table(title="Analyzers")
    table.add_column("order", justify="right")
    table.add_column("name", style="cyan")
    table.add_column("version")
    table.add_column("gpu")
    table.add_column("available")
    table.add_column("description", overflow="fold")
    for analyzer in all_analyzers():
        info = analyzer.describe()
        table.add_row(
            str(info["order"]),
            str(info["name"]),
            str(info["version"]),
            "yes" if info["needs_gpu"] else "",
            "[green]yes[/green]" if info["available"] else "[yellow]no[/yellow]",
            str(info["description"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Correction
# ---------------------------------------------------------------------------


@app.command()
def reanalyze(
    clip_ids: Optional[list[str]] = typer.Argument(None, help="Clip ids (default: all)."),
    analyzer: Optional[list[str]] = typer.Option(
        None, "--analyzer", "-a", help="Limit to these analyzers."
    ),
    missing: bool = typer.Option(
        False, "--missing", help="Only clips lacking a current result (after installing a backend)."
    ),
    process: bool = typer.Option(True, "--process/--queue-only", help="Run now, or just queue."),
) -> None:
    """Re-run AI analysis on selected clips."""
    pipeline = _pipeline()
    queued = pipeline.reanalyze(
        clip_ids=list(clip_ids) if clip_ids else None,
        analyzers=list(analyzer) if analyzer else None,
        only_missing=missing,
    )
    console.print(f"queued {queued} clip(s) for re-analysis")
    if queued and process:
        with console.status("[bold]analyzing[/bold]", spinner="dots"):
            summary = pipeline.run(scan=False, export=False, report=False)
        _print_run_summary(summary)


@app.command()
def recategorize() -> None:
    """Re-classify every clip from stored votes. No models are re-run."""
    pipeline = _pipeline()
    with console.status("[bold]recategorizing[/bold]", spinner="dots"):
        count = pipeline.recategorize()
    console.print(f"recategorized {count} clip(s)")


@app.command()
def resegment(
    source_id: str = typer.Argument(..., help="Source id or prefix."),
    min_score: Optional[float] = typer.Option(
        None, "--min-score", help="Override the boundary score threshold for this run."
    ),
) -> None:
    """Re-run segmentation on a source that split badly."""
    from sqlalchemy import select

    from .db.models import Source
    from .types import JobKind

    pipeline = _pipeline()
    settings = _settings()
    if min_score is not None:
        settings.segmentation.min_boundary_score = min_score
        console.print(f"using min_boundary_score={min_score}")

    with pipeline.db.session() as session:
        source = session.execute(
            select(Source).where(Source.id.like(f"{source_id}%")).limit(1)
        ).scalar_one_or_none()
        if source is None:
            error_console.print(f"no source matching {source_id!r}")
            raise typer.Exit(code=1)
        resolved_id = source.id
        filename = source.filename

    pipeline.queue.enqueue(
        JobKind.SEGMENT_SOURCE, {"source_id": resolved_id}, requeue_if_done=True
    )
    console.print(f"queued {filename} for re-segmentation")
    with console.status("[bold]segmenting[/bold]", spinner="dots"):
        summary = pipeline.run(scan=False, export=False, report=False)
    _print_run_summary(summary)


@app.command()
def retry() -> None:
    """Give dead jobs another chance."""
    pipeline = _pipeline()
    count = pipeline.queue.requeue_dead()
    console.print(f"requeued {count} dead job(s)")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: list[str] = typer.Argument(..., help='e.g. "every Taco Bell commercial"'),
    limit: int = typer.Option(25, "--limit", "-n"),
    include_alternates: bool = typer.Option(False, "--include-alternates"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Search the library in plain English."""
    from .search.service import SearchService

    pipeline = _pipeline()
    service = SearchService(_settings(), pipeline.db)
    response = service.search(
        " ".join(query), limit=limit, include_alternates=include_alternates
    )

    if as_json:
        console.print_json(json.dumps(response.as_dict(), default=str))
        return

    console.print(f"[dim]{response.explanation}[/dim]\n")
    if not response.hits:
        console.print("no matches")
        return

    table = Table(title=f"{len(response.hits)} result(s)")
    table.add_column("id", style="dim")
    table.add_column("category", style="cyan")
    table.add_column("dur", justify="right")
    table.add_column("network")
    table.add_column("year", justify="right")
    table.add_column("title", overflow="fold")
    table.add_column("via", style="dim")
    for hit in response.hits:
        clip = hit.clip
        table.add_row(
            hit.clip_id[:8],
            str(clip.category) if clip else "",
            format_duration(clip.duration_s) if clip else "",
            (clip.network or "-") if clip else "",
            str((clip.air_year or "-") if clip else ""),
            (clip.title or "-") if clip else "",
            ",".join(hit.matched_by),
        )
    console.print(table)


@app.command()
def similar(
    clip_id: str = typer.Argument(..., help="Clip id or prefix."),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """Find clips semantically similar to one you already have."""
    from .search.service import SearchService

    pipeline = _pipeline()
    service = SearchService(_settings(), pipeline.db)
    response = service.similar(clip_id, limit=limit)
    console.print(f"[dim]{response.explanation}[/dim]\n")
    for hit in response.hits:
        clip = hit.clip
        console.print(
            f"  {hit.score:.3f}  {hit.clip_id[:8]}  {clip.category if clip else ''}  "
            f"{clip.title if clip else ''}"
        )


@app.command()
def fetch(
    queries: Optional[list[str]] = typer.Argument(None, help="Search terms."),
    preset: Optional[list[str]] = typer.Option(
        None, "--preset", "-p", help="Named era preset (see --list-presets)."
    ),
    list_presets: bool = typer.Option(False, "--list-presets"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Items per query."),
    plan_only: bool = typer.Option(False, "--plan-only", help="Show what would be downloaded."),
) -> None:
    """Search Internet Archive and queue recordings for download."""
    from .sources.internet_archive import ERA_PRESETS
    from .sources.manager import DownloadManager

    if list_presets:
        table = Table(title="Internet Archive presets")
        table.add_column("preset", style="cyan")
        table.add_column("query", overflow="fold")
        for name, value in ERA_PRESETS.items():
            table.add_row(name, value)
        console.print(table)
        return

    terms = list(queries or [])
    for name in preset or []:
        if name not in ERA_PRESETS:
            error_console.print(f"unknown preset {name!r}; see --list-presets")
            raise typer.Exit(code=1)
        terms.append(ERA_PRESETS[name])

    pipeline = _pipeline()
    manager = DownloadManager(_settings(), pipeline.db, pipeline.queue)
    with console.status("[bold]searching archive.org[/bold]", spinner="dots"):
        plan = manager.plan(terms or None, limit=limit)

    console.print(plan.summary())
    for item in plan.queued[:20]:
        console.print(f"  {item['size_mb']:>8.0f} MB  {item['identifier']}  {item['title'][:60]}")

    if plan_only or not plan.queued:
        return
    console.print("\ndownloading (they will be queued for processing as they arrive)...")
    with console.status("[bold]downloading[/bold]", spinner="dots"):
        summary = pipeline.run(scan=True, export=False, report=False)
    _print_run_summary(summary)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@app.command()
def dashboard(
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Start the local dashboard."""
    settings = _settings()
    if not capabilities.has("fastapi"):
        error_console.print(
            escape("the dashboard needs: pip install 'broadcastbrain[dashboard]'")
        )
        raise typer.Exit(code=1)

    from .dashboard.app import serve

    resolved_host = host or settings.dashboard.host
    resolved_port = port or settings.dashboard.port
    url = f"http://{resolved_host}:{resolved_port}"
    console.print(f"[green]dashboard[/green] -> {url}")

    if open_browser:
        import webbrowser

        webbrowser.open(url)
    serve(settings, host=resolved_host, port=resolved_port)


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"{APP_NAME} {__version__}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_run_summary(summary: Any) -> None:
    data = summary.as_dict()
    workers = data.get("workers") or {}
    console.print(
        f"\n[green]done[/green] in {format_duration(float(data.get('duration_s') or 0))}: "
        f"{data.get('clips_created', 0)} clip(s) created, "
        f"{workers.get('completed', 0)} job(s) completed, "
        f"{workers.get('dead', 0)} failed"
    )
    discovery = data.get("discovery")
    if discovery:
        console.print(
            f"  discovery: {discovery.get('new_sources', 0)} new, "
            f"{discovery.get('already_known', 0)} known, "
            f"{discovery.get('rejected', 0)} rejected"
        )
    if data.get("export"):
        console.print(f"  export:    {data['export'].get('clips', 0)} clips -> {data['export'].get('root')}")
    if data.get("report"):
        console.print(f"  report:    {data['report']}")


def _starter_config(settings: Settings) -> str:
    """A commented starter config with the values this instance resolved to."""
    return f'''# {APP_NAME} configuration
# Every value here is optional -- these are the defaults, written out so you can
# see what exists. Environment variables override this file:
#   BB_WORKERS__CPU_WORKERS=4  overrides [workers] cpu_workers

instance_name = "{settings.instance_name}"
era_start_year = {settings.era_start_year}
era_end_year = {settings.era_end_year}

[paths]
# Only `root` is normally needed; everything else derives from it.
root = "{settings.paths.root}"

[database]
# Switch to PostgreSQL later without touching anything else:
# url = "postgresql+psycopg://user:pass@localhost/broadcastbrain"

[workers]
cpu_workers = 0      # 0 = auto (cores - 1)
gpu_workers = {settings.workers.gpu_workers}
low_memory = false
min_free_gb = {settings.workers.min_free_gb}

[segmentation]
# Lower min_boundary_score to cut more aggressively on soft VHS transitions.
min_boundary_score = {settings.segmentation.min_boundary_score}
min_segment_s = {settings.segmentation.min_segment_s}
reencode = false     # true = frame-accurate cuts, much slower

[analysis]
whisper_model = "{settings.analysis.whisper_model}"
frames_per_clip = {settings.analysis.frames_per_clip}
# faces_gallery = "/path/to/gallery"   # <gallery>/<Person Name>/*.jpg

[dedupe]
max_normalized_distance = {settings.dedupe.max_normalized_distance}

[export]
break_interval_s = {settings.export.break_interval_s}
pod_target_s = {settings.export.pod_target_s}

[archive]
enabled = false
queries = ["MTV 2004", "Discovery Channel recording"]
max_items_per_query = 25
max_total_gb = 500.0
'''


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except BroadcastBrainError as exc:
        error_console.print(f"error: {exc}")
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
