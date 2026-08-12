"""Re-analysis must carry all the way through to the library.

Found by re-running a real recording after changing a classification rule: the
analyzer results updated, and nothing else did. The clips kept their old
categories and their old library paths, because every downstream job already
existed in state ``done`` and a plain enqueue is a no-op for those.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from broadcastbrain.config import Settings
from broadcastbrain.db.base import Database
from broadcastbrain.db.models import Analysis, Clip, Job
from broadcastbrain.pipeline.orchestrator import Pipeline
from broadcastbrain.types import JobKind, JobState


@pytest.fixture()
def processed(settings: Settings, db: Database, fake_ffmpeg: Path, sample_source: Path) -> Pipeline:
    pipeline = Pipeline(settings, db=db)
    pipeline.initialize()
    pipeline.run(report=False)
    return pipeline


class TestReanalysisReachesTheLibrary:
    def _categorize_job(self, pipeline: Pipeline, clip_id: str) -> Job:
        with pipeline.db.session() as session:
            job = (
                session.query(Job)
                .filter(Job.kind == JobKind.CATEGORIZE_CLIP.value)
                .filter(Job.payload["clip_id"].as_string() == clip_id)
                .one_or_none()
            )
            assert job is not None
            session.expunge(job)
            return job

    def test_downstream_jobs_are_requeued(self, processed: Pipeline) -> None:
        with processed.db.session() as session:
            clip_id = str(session.query(Clip.id).first()[0])

        before = self._categorize_job(processed, clip_id)
        assert before.state == JobState.DONE.value, "precondition: the chain completed"
        assert before.finished_at is not None

        processed.reanalyze(clip_ids=[clip_id])
        processed.run(scan=False, export=False, report=False)

        after = self._categorize_job(processed, clip_id)
        # Requeuing resets the attempt counter, so the completion timestamp --
        # not the attempt count -- is what proves the stage ran again.
        assert after.state == JobState.DONE.value
        assert after.finished_at is not None
        assert after.finished_at > before.finished_at, (
            "the categorize stage never re-ran after re-analysis"
        )

    def test_a_rule_change_reaches_the_stored_category(self, processed: Pipeline) -> None:
        """The end-to-end property: change a vote, see the library change."""
        from broadcastbrain.analyze import get
        from broadcastbrain.analyze.base import AnalysisContext, AnalyzerResult
        from broadcastbrain.types import Category

        analyzer = get("estimation")
        assert analyzer is not None
        original_run = type(analyzer).run
        original_version = type(analyzer).version

        def forced(self: object, ctx: AnalysisContext) -> AnalyzerResult:
            result = AnalyzerResult(backend="test")
            result.vote(Category.MUSIC_VIDEO, 5.0, "forced by test")
            return result

        type(analyzer).run = forced  # type: ignore[method-assign]
        type(analyzer).version = "test-rule-change"  # type: ignore[assignment]
        try:
            processed.reanalyze(only_missing=True)
            processed.run(scan=False, export=False, report=False)
        finally:
            type(analyzer).run = original_run  # type: ignore[method-assign]
            type(analyzer).version = original_version  # type: ignore[assignment]

        with processed.db.session() as session:
            categories = {c.category for c in session.query(Clip).all()}
        assert Category.MUSIC_VIDEO.value in categories, (
            "a changed classification rule did not reach the library"
        )

    def test_reanalysis_does_not_recut_media(self, processed: Pipeline) -> None:
        with processed.db.session() as session:
            before = {c.id: (c.file_path, c.size_bytes) for c in session.query(Clip).all()}

        processed.reanalyze(only_missing=False)
        processed.run(scan=False, export=False, report=False)

        with processed.db.session() as session:
            after = {c.id: (c.file_path, c.size_bytes) for c in session.query(Clip).all()}
        assert before == after, "re-analysis must never move or re-cut clip media"

    def test_analyzer_version_bump_invalidates_only_that_analyzer(
        self, processed: Pipeline
    ) -> None:
        with processed.db.session() as session:
            rows = session.query(Analysis).filter(Analysis.analyzer == "technical").all()
            technical_before = {r.clip_id: r.created_at for r in rows}
        assert technical_before, "precondition: technical analyzer ran"

        # Re-run only estimation; technical results must be left untouched.
        processed.reanalyze(analyzers=["estimation"])
        processed.run(scan=False, export=False, report=False)

        with processed.db.session() as session:
            rows = session.query(Analysis).filter(Analysis.analyzer == "technical").all()
            technical_after = {r.clip_id: r.created_at for r in rows}
        assert technical_before == technical_after, "an unrelated analyzer was re-run"


class TestReanalyzeMissingUsesTheCache:
    """`--missing` must not re-run analyzers that are already current.

    It was passing force=["*"], which overrides the version cache and re-ran
    every analyzer on every selected clip -- re-transcribing audio that already
    had a transcript. The flag promises the opposite.
    """

    def test_missing_does_not_force_everything(self, processed: Pipeline) -> None:
        from broadcastbrain.db.models import Job

        processed.reanalyze(only_missing=True)
        with processed.db.session() as session:
            payloads = [
                job.payload
                for job in session.query(Job)
                .filter(Job.kind == JobKind.ANALYZE_CLIP.value)
                .all()
            ]
        assert payloads, "precondition: analyze jobs exist"
        assert all("*" not in (p.get("force") or []) for p in payloads), (
            "--missing forced a full re-analysis instead of using the cache"
        )

    def test_explicit_analyzer_is_still_forced(self, processed: Pipeline) -> None:
        from broadcastbrain.db.models import Job

        processed.reanalyze(analyzers=["estimation"])
        with processed.db.session() as session:
            job = (
                session.query(Job)
                .filter(Job.kind == JobKind.ANALYZE_CLIP.value)
                .first()
            )
        assert job is not None
        assert job.payload.get("force") == ["estimation"]

    def test_bare_reanalyze_still_forces_everything(self, processed: Pipeline) -> None:
        from broadcastbrain.db.models import Job

        processed.reanalyze()
        with processed.db.session() as session:
            job = (
                session.query(Job)
                .filter(Job.kind == JobKind.ANALYZE_CLIP.value)
                .first()
            )
        assert job is not None
        assert job.payload.get("force") == ["*"]
