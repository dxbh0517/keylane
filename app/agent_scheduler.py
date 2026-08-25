"""Background agent loop — Hermes-style cron ticks inside the gateway.

Every minute the scheduler looks at due standing goals, runs each in an
isolated assistant session, and surfaces noteworthy results via the activity
bus and desktop notifications. Quiet ticks (``[SILENT]``) leave no toast.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from app.activity import get_activity_bus
from app.agent_goals import SILENT_MARKER, get_goal_store, is_silent_result
from app.assistant import get_assistant
from app.assistant_settings import load_assistant_settings
from app.memory_store import get_memory_store
from app.sessions import get_session_store

logger = logging.getLogger(__name__)

TICK_SECONDS = 60


class AgentScheduler:
    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._running_goal: str | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="keylane-agent-scheduler")
        logger.info("Agent scheduler started (tick=%ss).", TICK_SECONDS)

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("Agent scheduler stopped.")

    async def _loop(self) -> None:
        # Stagger the first tick so startup MCP discovery can finish.
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=5.0)
            return
        except TimeoutError:
            pass
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception:  # noqa: BLE001
                logger.exception("Agent scheduler tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=TICK_SECONDS)
                return
            except TimeoutError:
                continue

    async def tick(self) -> list[dict[str, Any]]:
        settings = load_assistant_settings()
        if not settings.agent.enabled:
            return []
        store = get_goal_store()
        due = store.due()
        results: list[dict[str, Any]] = []
        for goal in due:
            if self._stop.is_set():
                break
            try:
                result = await self.run_goal(goal.id)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Goal %s failed", goal.id)
                store.mark_ran(
                    goal.id,
                    result=f"error: {exc}",
                    noteworthy=False,
                )
                results.append(
                    {"goal_id": goal.id, "ok": False, "error": str(exc), "noteworthy": False}
                )
        return results

    async def run_goal(self, goal_id: str) -> dict[str, Any]:
        store = get_goal_store()
        goal = store.get(goal_id)
        if goal is None or not goal.enabled:
            return {"goal_id": goal_id, "ok": False, "error": "unknown or disabled"}

        self._running_goal = goal_id
        activity = get_activity_bus()
        await activity.note(
            "notice",
            f"Agent: {goal.title or goal.id}",
            detail=goal.instruction[:200],
        )

        memory = get_memory_store()
        memory_block = memory.prompt_block(goal.instruction)
        prompt = (
            f"[Standing goal: {goal.title or goal.id}]\n"
            f"Kind: {goal.kind}. Interval: every {goal.interval_seconds}s.\n\n"
            f"{goal.instruction.strip()}\n\n"
            "This is an unattended check. Use only the tools you need. "
            f"If nothing needs the user's attention, reply with a final answer "
            f"containing exactly {SILENT_MARKER} and nothing else noteworthy. "
            "If something matters, produce a short canvas that proposes the next "
            "helpful action and ask before drafting a reply or creating a calendar "
            "event.\n"
        )
        if memory_block:
            prompt += f"\n{memory_block}\n"

        assistant = get_assistant()
        # Auto-confirm only the explicitly allowlisted read tools for this goal.
        confirmed = set(goal.auto_confirm)
        # Sensible defaults for mail/calendar reads.
        if goal.kind == "email":
            confirmed.update(
                n
                for n in assistant.tools.all_names()
                if n.startswith("mailspring.")
                and any(x in n for x in ("search", "list", "get", "read"))
            )
        if goal.kind == "calendar":
            confirmed.update(
                n
                for n in assistant.tools.all_names()
                if ("calendar" in n or "event" in n)
                and any(x in n for x in ("list", "upcoming", "get"))
            )

        try:
            outcome = await assistant.run(
                prompt,
                confirmed_tools=confirmed,
                history="",  # fresh session — Hermes isolation
                agent_mode=True,
            )
        finally:
            self._running_goal = None

        answer = (outcome.answer or outcome.question or "").strip()
        noteworthy = not is_silent_result(answer) and not outcome.error
        store.mark_ran(goal_id, result=answer or (outcome.error or ""), noteworthy=noteworthy)

        payload = {
            "goal_id": goal_id,
            "title": goal.title,
            "ok": not outcome.error,
            "noteworthy": noteworthy,
            "answer": answer,
            "canvas": outcome.canvas,
            "error": outcome.error,
        }

        if noteworthy:
            session_store = get_session_store()
            session = session_store.get_or_create(goal.origin_session_id)
            session_store.append_assistant(
                session,
                answer,
                canvas=outcome.canvas,
                tools_used=[s.tool for s in outcome.steps if s.tool],
            )
            payload["session_id"] = session.session_id
            await activity.note(
                "alert",
                goal.title or "Keylane agent",
                detail=answer[:280],
            )
            _desktop_notify(goal.title or "Keylane", answer[:280])

        return payload

    def status(self) -> dict[str, Any]:
        settings = load_assistant_settings()
        goals = get_goal_store().list()
        return {
            "enabled": settings.agent.enabled,
            "running": self.running,
            "current_goal": self._running_goal,
            "goal_count": len(goals),
            "enabled_goals": sum(1 for g in goals if g.enabled),
            "goals": [
                {
                    "id": g.id,
                    "title": g.title,
                    "kind": g.kind,
                    "enabled": g.enabled,
                    "interval_seconds": g.interval_seconds,
                    "next_run_at": g.next_run_at.isoformat() if g.next_run_at else None,
                    "last_run_at": g.last_run_at.isoformat() if g.last_run_at else None,
                    "last_noteworthy": g.last_noteworthy,
                }
                for g in goals
            ],
        }


def _desktop_notify(title: str, body: str) -> None:
    try:
        import shutil
        import subprocess

        if not shutil.which("notify-send"):
            return
        subprocess.Popen(  # noqa: S603
            ["notify-send", "--app-name=Keylane", title[:80], body[:280]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        logger.debug("notify-send failed", exc_info=True)


_scheduler: AgentScheduler | None = None


def get_agent_scheduler() -> AgentScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AgentScheduler()
    return _scheduler
