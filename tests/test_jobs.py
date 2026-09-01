"""Background jobs: ids, status, cancellation, and the depth cap."""

from __future__ import annotations

import time

import pytest

from seams.errors import JobError
from seams.jobs import MAX_DEPTH, Job, JobRegistry


@pytest.fixture()
def jobs():
    registry = JobRegistry()
    yield registry
    registry.shutdown()


def _settle(registry: JobRegistry, job: Job, timeout: float = 3.0) -> dict:
    return registry.output(job.id, wait=True, timeout_seconds=timeout)


def test_a_started_job_has_an_id_and_a_status(jobs) -> None:
    job = jobs.start(kind="agent", label="research", work=lambda j: "done")
    assert job.id
    assert _settle(jobs, job)["status"] == "completed"


def test_the_result_is_readable_after_it_finishes(jobs) -> None:
    job = jobs.start(kind="agent", label="research", work=lambda j: "the answer")
    assert _settle(jobs, job)["output"] == "the answer"


def test_a_running_job_reads_without_blocking(jobs) -> None:
    job = jobs.start(kind="agent", label="slow", work=lambda j: (time.sleep(0.4), "late")[1])
    payload = jobs.output(job.id)
    assert payload["status"] == "running"
    assert "do not poll" in payload["note"]


def test_a_failing_job_reports_its_error(jobs) -> None:
    def _boom(job: Job) -> str:
        raise RuntimeError("searxng is down")

    job = jobs.start(kind="agent", label="bad", work=_boom)
    payload = _settle(jobs, job)
    assert payload["status"] == "failed"
    assert payload["error"] == "searxng is down"


def test_listing_shows_every_job(jobs) -> None:
    first = jobs.start(kind="agent", label="a", work=lambda j: "a")
    second = jobs.start(kind="agent", label="b", work=lambda j: "b")
    _settle(jobs, first)
    _settle(jobs, second)
    assert {j["job_id"] for j in jobs.list()} == {first.id, second.id}


def test_a_killed_job_stops_cooperatively(jobs) -> None:
    def _loop(job: Job) -> str:
        for _ in range(200):
            if job.cancel.is_set():
                return "stopped early"
            time.sleep(0.01)
        return "ran to completion"

    job = jobs.start(kind="agent", label="loop", work=_loop)
    time.sleep(0.05)
    accepted = jobs.kill(job.id)
    assert accepted["status"] == "stopping"

    payload = _settle(jobs, job)
    assert payload["status"] == "killed"
    assert payload["output"] == "stopped early"


def test_killing_a_finished_job_is_accepted(jobs) -> None:
    job = jobs.start(kind="agent", label="quick", work=lambda j: "done")
    _settle(jobs, job)
    assert jobs.kill(job.id)["note"] == "already finished"


def test_reading_an_unknown_job_names_the_id(jobs) -> None:
    with pytest.raises(JobError) as exc:
        jobs.output("nosuchid")
    assert exc.value.code == "JOB_UNKNOWN"


def test_a_wait_is_capped_and_leaves_the_job_alive(jobs) -> None:
    job = jobs.start(kind="agent", label="slow", work=lambda j: (time.sleep(1.0), "late")[1])
    payload = jobs.output(job.id, wait=True, timeout_seconds=0.05)
    assert payload["status"] == "running"


def test_background_work_cannot_nest_without_end(jobs) -> None:
    """A background agent starting background agents is otherwise unbounded."""
    depths: list[int] = []

    def _spawn(job: Job) -> str:
        from seams.jobs import current_depth

        depths.append(current_depth())
        try:
            child = jobs.start(kind="agent", label="deeper", work=_spawn)
        except JobError as exc:
            return exc.code
        return _settle(jobs, child).get("output", "")

    top = jobs.start(kind="agent", label="top", work=_spawn)
    assert _settle(jobs, top, timeout=5.0)["output"] == "JOB_DEPTH_EXCEEDED"
    assert max(depths) == MAX_DEPTH


def test_the_view_never_leaks_internal_handles(jobs) -> None:
    job = jobs.start(kind="agent", label="x", work=lambda j: "x")
    _settle(jobs, job)
    assert set(jobs.list()[0]) == {"job_id", "kind", "label", "status", "seconds"}
