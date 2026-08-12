"""The durable queue: idempotence, atomic claiming, leases and retries.

These behaviours are the difference between "resumable" and "hopefully
resumable", so they are tested against a real SQLite database rather than a
mock.
"""

from __future__ import annotations

import threading
from datetime import timedelta

from broadcastbrain.db.models import Job, utcnow
from broadcastbrain.db.queue import JobQueue, default_dedupe_key
from broadcastbrain.types import JobKind, JobState


class TestEnqueue:
    def test_returns_an_id(self, queue: JobQueue) -> None:
        assert queue.enqueue(JobKind.INGEST_SOURCE, {"source_id": "abc"}) is not None

    def test_identical_work_is_not_queued_twice(self, queue: JobQueue) -> None:
        queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"}) is None
        assert queue.pending_count() == 1

    def test_non_identity_payload_keys_do_not_split_the_job(self, queue: JobQueue) -> None:
        queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1", "reason": "first"})
        assert queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1", "reason": "second"}) is None

    def test_different_clips_are_different_jobs(self, queue: JobQueue) -> None:
        assert queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "a"}) is not None
        assert queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "b"}) is not None

    def test_completed_work_is_not_repeated(self, queue: JobQueue) -> None:
        job_id = queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert job_id is not None
        queue.claim("w1")
        queue.complete(job_id)
        assert queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"}) is None

    def test_requeue_if_done_forces_a_rerun(self, queue: JobQueue) -> None:
        job_id = queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert job_id is not None
        queue.claim("w1")
        queue.complete(job_id)
        assert queue.enqueue(
            JobKind.ANALYZE_CLIP, {"clip_id": "c1"}, requeue_if_done=True
        ) == job_id

    def test_dedupe_key_is_stable(self) -> None:
        first = default_dedupe_key(JobKind.ANALYZE_CLIP, {"clip_id": "x", "extra": 1})
        second = default_dedupe_key(JobKind.ANALYZE_CLIP, {"clip_id": "x", "extra": 2})
        assert first == second


class TestClaim:
    def test_priority_order(self, queue: JobQueue) -> None:
        queue.enqueue(JobKind.EXPORT_LIBRARY, {"scope": "all"})       # priority 5
        queue.enqueue(JobKind.INGEST_SOURCE, {"source_id": "s1"})     # priority 50
        job = queue.claim("w1")
        assert job is not None and job.kind == JobKind.INGEST_SOURCE.value

    def test_a_job_is_claimed_once(self, queue: JobQueue) -> None:
        queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert queue.claim("w1") is not None
        assert queue.claim("w2") is None

    def test_concurrent_claims_never_double_book(self, queue: JobQueue) -> None:
        """The property everything else depends on: no clip processed twice."""
        for index in range(20):
            queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": f"c{index}"})

        claimed: list[int] = []
        lock = threading.Lock()

        def worker(name: str) -> None:
            while True:
                job = queue.claim(name)
                if job is None:
                    return
                with lock:
                    claimed.append(job.id)

        threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(claimed) == 20
        assert len(set(claimed)) == 20, "a job was claimed by two workers"

    def test_gpu_partitioning(self, queue: JobQueue) -> None:
        queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "gpu"}, needs_gpu=True)
        assert queue.claim("cpu-1", allow_gpu=False) is None
        assert queue.claim("gpu-1", gpu_only=True) is not None

    def test_kind_filter(self, queue: JobQueue) -> None:
        queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert queue.claim("w1", kinds=[JobKind.EXPORT_LIBRARY]) is None
        assert queue.claim("w1", kinds=[JobKind.ANALYZE_CLIP]) is not None

    def test_future_jobs_are_not_claimed(self, queue: JobQueue) -> None:
        queue.enqueue(
            JobKind.ANALYZE_CLIP, {"clip_id": "later"}, available_at=utcnow() + timedelta(hours=1)
        )
        assert queue.claim("w1") is None


class TestFailureHandling:
    def test_transient_failure_is_retried_with_backoff(self, queue: JobQueue) -> None:
        job_id = queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert job_id is not None
        queue.claim("w1")
        assert queue.fail(job_id, "disk busy") is JobState.PENDING
        # Backoff means it is not immediately claimable again.
        assert queue.claim("w2") is None

    def test_permanent_failure_goes_straight_to_dead(self, queue: JobQueue) -> None:
        job_id = queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert job_id is not None
        queue.claim("w1")
        assert queue.fail(job_id, "file is not video", permanent=True) is JobState.DEAD

    def test_retries_are_exhausted(self, queue: JobQueue) -> None:
        job_id = queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"}, max_attempts=2)
        assert job_id is not None
        with queue.db.write_session() as session:
            job = session.get(Job, job_id)
            assert job is not None
            job.attempts = 2
        assert queue.fail(job_id, "still broken") is JobState.DEAD

    def test_dead_jobs_can_be_requeued(self, queue: JobQueue) -> None:
        job_id = queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert job_id is not None
        queue.claim("w1")
        queue.fail(job_id, "bad", permanent=True)
        assert queue.requeue_dead() == 1
        assert queue.claim("w2") is not None


class TestLeases:
    def test_expired_lease_is_reclaimed(self, queue: JobQueue) -> None:
        """A worker that dies must not strand its job forever."""
        job_id = queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert job_id is not None
        queue.claim("dead-worker")

        with queue.db.write_session() as session:
            job = session.get(Job, job_id)
            assert job is not None
            job.lease_expires_at = utcnow() - timedelta(minutes=5)

        assert queue.reclaim_expired() == 1
        assert queue.claim("healthy-worker") is not None

    def test_live_lease_is_left_alone(self, queue: JobQueue) -> None:
        queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        queue.claim("w1")
        assert queue.reclaim_expired() == 0

    def test_heartbeat_extends_the_lease_and_reports_progress(self, queue: JobQueue) -> None:
        job_id = queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert job_id is not None
        queue.claim("w1")
        queue.heartbeat(job_id, progress=0.5, note="transcribing")

        with queue.db.session() as session:
            job = session.get(Job, job_id)
            assert job is not None
            assert job.progress == 0.5
            assert job.progress_note == "transcribing"
            assert job.lease_expires_at is not None and job.lease_expires_at > utcnow()


class TestIntrospection:
    def test_counts_cover_every_state(self, queue: JobQueue) -> None:
        queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        counts = queue.counts_by_state()
        assert counts["pending"] == 1
        assert set(counts) == {state.value for state in JobState}

    def test_has_runnable_work(self, queue: JobQueue) -> None:
        assert queue.has_runnable_work() is False
        queue.enqueue(JobKind.ANALYZE_CLIP, {"clip_id": "c1"})
        assert queue.has_runnable_work() is True
