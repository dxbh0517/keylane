"""Activity bus — what the gateway is doing right now.

The tray indicator, the popup and the control panel all need the same answer to
"is the assistant busy?". They get it from here: an in-memory ledger of live
tasks plus a fan-out of events over Server-Sent Events, with a plain snapshot
endpoint for clients that would rather poll.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_HISTORY = 40
MAX_QUEUE = 64

# Task states that still count as "the assistant is working".
BUSY_STATES = {"pending", "routing", "running", "verifying", "retrying", "thinking"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ActivityEvent(BaseModel):
    type: str
    """task | tool | step | notice"""

    task_id: str = ""
    status: str = ""
    title: str = ""
    detail: str = ""
    worker: str | None = None
    tool: str | None = None
    progress: float | None = None
    at: str = Field(default_factory=_now)


class ActivityTask(BaseModel):
    task_id: str
    title: str
    status: str = "pending"
    worker: str | None = None
    step: str = ""
    started_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    finished_at: str | None = None
    error: str | None = None

    @property
    def busy(self) -> bool:
        return self.status in BUSY_STATES


class ActivitySnapshot(BaseModel):
    busy: bool = False
    active_count: int = 0
    needs_attention: int = 0
    active: list[ActivityTask] = Field(default_factory=list)
    recent: list[ActivityTask] = Field(default_factory=list)
    at: str = Field(default_factory=_now)


class ActivityBus:
    def __init__(self) -> None:
        self._tasks: dict[str, ActivityTask] = {}
        self._history: deque[ActivityTask] = deque(maxlen=MAX_HISTORY)
        self._subscribers: set[asyncio.Queue[ActivityEvent]] = set()
        self._lock = asyncio.Lock()

    # -------------------------------------------------------------- emitting

    def _publish(self, event: ActivityEvent) -> None:
        dead: list[asyncio.Queue[ActivityEvent]] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled listener must not block the gateway.
                dead.append(queue)
        for queue in dead:
            self._subscribers.discard(queue)

    async def start_task(
        self, task_id: str, title: str, *, worker: str | None = None
    ) -> ActivityTask:
        async with self._lock:
            task = ActivityTask(
                task_id=task_id, title=title[:160], status="routing", worker=worker
            )
            self._tasks[task_id] = task
            self._publish(
                ActivityEvent(
                    type="task",
                    task_id=task_id,
                    status=task.status,
                    title=task.title,
                    worker=worker,
                )
            )
            return task

    async def update_task(
        self,
        task_id: str,
        *,
        status: str | None = None,
        worker: str | None = None,
        step: str | None = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            if status:
                task.status = status
            if worker:
                task.worker = worker
            if step is not None:
                task.step = step[:200]
            if error is not None:
                task.error = error[:500]
            task.updated_at = _now()

            finished = task.status in {"completed", "failed", "cancelled"}
            if finished:
                task.finished_at = _now()
                self._history.appendleft(task)
                self._tasks.pop(task_id, None)

            self._publish(
                ActivityEvent(
                    type="task",
                    task_id=task_id,
                    status=task.status,
                    title=task.title,
                    detail=task.step,
                    worker=task.worker,
                )
            )

    async def note(
        self,
        kind: str,
        title: str,
        *,
        detail: str = "",
        task_id: str = "",
        tool: str | None = None,
    ) -> None:
        async with self._lock:
            if task_id and task_id in self._tasks:
                self._tasks[task_id].step = title[:200]
                self._tasks[task_id].updated_at = _now()
            self._publish(
                ActivityEvent(
                    type=kind,
                    task_id=task_id,
                    title=title[:200],
                    detail=detail[:600],
                    tool=tool,
                )
            )

    # ------------------------------------------------------------- observing

    def snapshot(self) -> ActivitySnapshot:
        active = [t for t in self._tasks.values() if t.busy]
        waiting = [
            t for t in self._tasks.values() if t.status == "waiting_confirmation"
        ]
        return ActivitySnapshot(
            busy=bool(active),
            active_count=len(active),
            needs_attention=len(waiting),
            active=sorted(active + waiting, key=lambda t: t.started_at),
            recent=list(self._history)[:10],
        )

    @contextlib.asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[ActivityEvent]]:
        queue: asyncio.Queue[ActivityEvent] = asyncio.Queue(maxsize=MAX_QUEUE)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    async def sse_stream(
        self, stop: asyncio.Event | None = None
    ) -> AsyncIterator[str]:
        """Yield SSE frames: a snapshot first, then events, with keepalives.

        ``stop`` must be set when the gateway is shutting down. Without it this
        generator never returns, uvicorn's graceful shutdown waits on it
        forever, and systemd eventually SIGABRTs the process — the tray holds
        one of these streams open for the whole session.
        """
        async with self.subscribe() as queue:
            yield _sse("snapshot", self.snapshot().model_dump_json())

            while True:
                if stop is not None and stop.is_set():
                    return

                getter = asyncio.ensure_future(queue.get())
                waiters: set[asyncio.Future] = {getter}
                stopper = asyncio.ensure_future(stop.wait()) if stop else None
                if stopper is not None:
                    waiters.add(stopper)

                try:
                    done, _pending = await asyncio.wait(
                        waiters,
                        timeout=20.0,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    # Never leave the losing waiter dangling on the queue.
                    for task in waiters:
                        if not task.done():
                            task.cancel()

                if stopper is not None and stopper in done:
                    getter.cancel()
                    return
                if getter not in done:
                    yield ": keepalive\n\n"
                    continue

                event = getter.result()
                yield _sse("event", event.model_dump_json())
                yield _sse("snapshot", self.snapshot().model_dump_json())


def _sse(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


_bus: ActivityBus | None = None


def get_activity_bus() -> ActivityBus:
    global _bus
    if _bus is None:
        _bus = ActivityBus()
    return _bus


def snapshot_dict() -> dict[str, Any]:
    return get_activity_bus().snapshot().model_dump()
