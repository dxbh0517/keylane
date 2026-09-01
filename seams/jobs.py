"""Background jobs, kind-agnostic.

`run_background` used to spawn a whole agent in a thread and drop its answer in
the inbox: no id, no status, no way to stop it, and no depth guard — so a
background agent could start another one, forever.

Everything that runs out of band now goes through one registry with one set of
controls. A background agent run and a delegated subagent are the same kind of
thing to the model: something with an id, that it can read from and stop.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from seams.errors import JobError

logger = logging.getLogger(__name__)

JobStatus = Literal["running", "completed", "failed", "killed"]

# How deep background work may nest. A background agent may delegate once; its
# child may not delegate again. Without a cap a single prompt can fan out until
# the machine gives up.
MAX_DEPTH = 2

DEFAULT_WAIT_SECONDS = 30.0
MAX_WAIT_SECONDS = 300.0


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: JobStatus = "running"
    depth: int = 0
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    output: str = ""
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None

    def view(self) -> dict[str, Any]:
        """What the model is shown. No handles, no futures."""
        return {
            "job_id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "seconds": round((self.finished_at or time.time()) - self.created_at, 1),
        }


class _Depth(threading.local):
    value: int = 0


_depth = _Depth()


def current_depth() -> int:
    """How deep the calling thread already is in background work."""
    return getattr(_depth, "value", 0)


class JobRegistry:
    """Starts, reports on, and stops background work of any kind."""

    def __init__(self, max_workers: int = 4) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="keylane-job")

    def start(
        self,
        *,
        kind: str,
        label: str,
        work: Callable[[Job], str],
    ) -> Job:
        """Run `work` in the background. It receives its own Job for cancellation."""
        depth = current_depth()
        if depth >= MAX_DEPTH:
            raise JobError(
                "JOB_DEPTH_EXCEEDED",
                f"background work is already {depth} levels deep; do this one directly "
                "instead of starting another background job",
            )

        job = Job(id=uuid.uuid4().hex[:8], kind=kind, label=label, depth=depth + 1)
        with self._lock:
            self._jobs[job.id] = job

        def _runner() -> None:
            _depth.value = job.depth
            try:
                result = work(job)
                job.output = result if isinstance(result, str) else str(result)
                job.status = "killed" if job.cancel.is_set() else "completed"
            except Exception as exc:  # noqa: BLE001
                logger.exception("job %s (%s) failed", job.id, job.kind)
                job.error = str(exc)
                job.status = "failed"
            finally:
                job.finished_at = time.time()
                job.done.set()
                _depth.value = 0

        job.future = self._pool.submit(_runner)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobError("JOB_UNKNOWN", f"no background job with id {job_id!r}")
        return job

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at)
        return [job.view() for job in jobs]

    def output(
        self,
        job_id: str,
        *,
        wait: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Read a job. Non-blocking unless `wait`, which is capped."""
        job = self.get(job_id)
        if wait and not job.done.is_set():
            budget = min(
                timeout_seconds if timeout_seconds is not None else DEFAULT_WAIT_SECONDS,
                MAX_WAIT_SECONDS,
            )
            job.done.wait(timeout=budget)

        payload = job.view()
        if job.status == "running":
            # Say what to do rather than returning an empty string: the model
            # should keep working, not spin on this job.
            payload["note"] = (
                "Still running. You are told when it finishes — do not poll it; "
                "carry on with something else."
            )
        elif job.status == "failed":
            payload["error"] = job.error
        else:
            payload["output"] = job.output
        return payload

    def kill(self, job_id: str, reason: str = "") -> dict[str, Any]:
        """Ask a job to stop. Returns at once; it settles when the work does."""
        job = self.get(job_id)
        if job.done.is_set():
            return {"job_id": job.id, "status": job.status, "note": "already finished"}
        job.cancel.set()
        if reason:
            job.error = reason
        return {
            "job_id": job.id,
            "status": "stopping",
            "note": "The stop request was accepted; the job settles once its work stops.",
        }

    def shutdown(self) -> None:
        for job in list(self._jobs.values()):
            job.cancel.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
