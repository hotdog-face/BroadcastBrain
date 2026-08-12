"""End-to-end pipeline tests against a stub ffmpeg.

These exercise the real orchestrator, the real queue, the real database and the
real stage handlers -- only the media decoding is faked. That means the
properties the brief actually asks for are verified rather than asserted:
resumability, never reprocessing, nothing discarded, and a usable export at the
end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from broadcastbrain.config import Settings
from broadcastbrain.db.base import Database
from broadcastbrain.db.models import Clip
from broadcastbrain.db.repo import ClipRepository, SourceRepository
from broadcastbrain.pipeline.orchestrator import Pipeline
from broadcastbrain.types import ClipStatus, SourceStatus, Stage


@pytest.fixture()
def pipeline(settings: Settings, db: Database, fake_ffmpeg: Path) -> Pipeline:
    instance = Pipeline(settings, db=db)
    instance.initialize()
    return instance


class TestDiscovery:
    def test_new_file_is_registered_and_queued(
        self, pipeline: Pipeline, sample_source: Path
    ) -> None:
        result = pipeline.scan_incoming()
        assert result.new_sources == 1
        assert pipeline.queue.pending_count() == 1

    def test_rescanning_does_not_duplicate(self, pipeline: Pipeline, sample_source: Path) -> None:
        pipeline.scan_incoming()
        second = pipeline.scan_incoming()
        assert second.new_sources == 0
        assert second.already_known == 1

    def test_unsupported_files_are_moved_not_deleted(
        self, pipeline: Pipeline, settings: Settings
    ) -> None:
        junk = settings.paths.require("incoming") / "notes.txt"
        junk.write_text("this is not television")

        result = pipeline.scan_incoming()

        assert result.rejected == 1
        assert not junk.exists(), "rejected file should have been moved out of Incoming"
        moved = settings.paths.require("rejected") / "notes.txt"
        assert moved.exists(), "rejected file must be kept, never deleted"
        assert moved.with_suffix(".txt.reason.txt").exists(), "rejection reason should be recorded"

    def test_tiny_files_are_rejected(self, pipeline: Pipeline, settings: Settings) -> None:
        stub = settings.paths.require("incoming") / "truncated.mp4"
        stub.write_bytes(b"\x00" * 32)
        assert pipeline.scan_incoming().rejected == 1


class TestFullRun:
    def test_a_source_becomes_a_categorized_library(
        self, pipeline: Pipeline, sample_source: Path, settings: Settings
    ) -> None:
        summary = pipeline.run(report=False)

        assert summary.clips_created > 0
        with pipeline.db.session() as session:
            source = SourceRepository.by_path(session, sample_source)
            assert source is not None
            assert source.status == SourceStatus.COMPLETE.value
            clips = ClipRepository.by_source(session, source.id)

        assert len(clips) > 1, "the fake detector output describes several boundaries"
        assert all(clip.stage == Stage.MATERIALIZED.value for clip in clips)
        assert all(clip.status == ClipStatus.READY.value for clip in clips)
        assert all(clip.file_path and Path(clip.file_path).exists() for clip in clips)

    def test_clips_land_in_category_folders(
        self, pipeline: Pipeline, sample_source: Path, settings: Settings
    ) -> None:
        pipeline.run(report=False)
        library = settings.paths.require("library")
        media = [p for p in library.rglob("*") if p.is_file() and p.suffix in {".mkv", ".ts", ".mp4"}]
        assert media, "no clips were materialized into the library"
        # Each clip is filed under a category directory, not dumped at the root.
        assert all(p.parent != library for p in media)

    def test_metadata_sidecar_is_written_for_every_clip(
        self, pipeline: Pipeline, sample_source: Path, settings: Settings
    ) -> None:
        pipeline.run(report=False)
        with pipeline.db.session() as session:
            clips = list(session.query(Clip).all())
        sidecars = list(settings.paths.require("metadata").glob("*.json"))
        assert len(sidecars) == len(clips)

    def test_export_is_produced(self, pipeline: Pipeline, sample_source: Path, settings: Settings) -> None:
        pipeline.run(report=False)
        export_root = settings.paths.require("export")
        for name in ("metadata.json", "channels.json", "schedule.json", "README.md"):
            assert (export_root / name).exists(), f"{name} missing from the export"

    def test_report_is_written(self, pipeline: Pipeline, sample_source: Path, settings: Settings) -> None:
        summary = pipeline.run(report=True)
        assert summary.report_path is not None
        assert Path(summary.report_path).exists()
        assert (settings.paths.require("logs") / "reports" / "latest.json").exists()

    def test_nothing_is_reprocessed_on_a_second_run(
        self, pipeline: Pipeline, sample_source: Path
    ) -> None:
        """The core promise: running again is nearly free and changes nothing."""
        first = pipeline.run(report=False)
        with pipeline.db.session() as session:
            clip_ids_before = {c.id for c in session.query(Clip).all()}

        second = pipeline.run(report=False)

        with pipeline.db.session() as session:
            clip_ids_after = {c.id for c in session.query(Clip).all()}

        assert second.clips_created == 0
        assert clip_ids_before == clip_ids_after, "clip identities must be stable across runs"
        assert first.clips_created > 0


class TestResumability:
    def test_an_interrupted_run_continues_where_it_stopped(
        self, pipeline: Pipeline, sample_source: Path
    ) -> None:
        # Simulate a crash: ingest and segment, then stop before analysis.
        pipeline.scan_incoming()
        job = pipeline.queue.claim("test-worker")
        assert job is not None

        from broadcastbrain.pipeline.stages import StageContext, get_handler

        ctx = StageContext(
            settings=pipeline.settings,
            db=pipeline.db,
            queue=pipeline.queue,
            progress=lambda _f, _m: None,
        )
        handler = get_handler(job.kind)
        assert handler is not None
        handler(ctx, job)
        pipeline.queue.complete(job.id)

        # A fresh pipeline object stands in for a restarted process.
        resumed = Pipeline(pipeline.settings, db=pipeline.db)
        summary = resumed.run(report=False)

        with resumed.db.session() as session:
            source = SourceRepository.by_path(session, sample_source)
            assert source is not None
            assert source.status == SourceStatus.COMPLETE.value
        assert summary.clips_created > 0

    def test_stranded_in_flight_work_is_reclaimed(
        self, pipeline: Pipeline, sample_source: Path
    ) -> None:
        from datetime import timedelta

        from broadcastbrain.db.models import Job, utcnow

        pipeline.scan_incoming()
        job = pipeline.queue.claim("worker-that-died")
        assert job is not None
        with pipeline.db.write_session() as session:
            stranded = session.get(Job, job.id)
            assert stranded is not None
            stranded.lease_expires_at = utcnow() - timedelta(hours=2)

        assert pipeline.recover() == 1
        summary = pipeline.run(report=False)
        assert summary.clips_created > 0


class TestReanalysis:
    def test_reanalyze_queues_clips_without_recutting_them(
        self, pipeline: Pipeline, sample_source: Path
    ) -> None:
        pipeline.run(report=False)
        with pipeline.db.session() as session:
            clips = list(session.query(Clip).all())
            paths_before = {c.id: c.file_path for c in clips}

        queued = pipeline.reanalyze(only_missing=False)
        assert queued == len(clips)

        pipeline.run(scan=False, export=False, report=False)
        with pipeline.db.session() as session:
            paths_after = {c.id: c.file_path for c in session.query(Clip).all()}
        assert paths_before == paths_after, "re-analysis must not move or re-cut media"

    def test_recategorize_runs_without_any_analyzer(
        self, pipeline: Pipeline, sample_source: Path
    ) -> None:
        pipeline.run(report=False)
        with pipeline.db.session() as session:
            expected = session.query(Clip).count()
        assert pipeline.recategorize() == expected


class TestStatus:
    def test_status_reports_the_library(self, pipeline: Pipeline, sample_source: Path) -> None:
        pipeline.run(report=False)
        snapshot = pipeline.status()
        assert snapshot["clips"]["total_clips"] > 0
        assert "queue" in snapshot
        assert snapshot["sources"]["complete"] == 1
