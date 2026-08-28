"""Background jobs and scheduled tasks."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from memory.store import get_store

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
        _scheduler.start()
    return _scheduler


def _run_agent_prompt(prompt: str, *, notify: bool = True) -> None:
    import asyncio

    from agent.loop import AIAgent

    async def _inner() -> str:
        agent = AIAgent()
        result = await agent.run(prompt)
        return result.answer

    try:
        answer = asyncio.run(_inner())
        if notify:
            from notify.desktop import send_notification
            from notify.tts_gate import maybe_speak_notify

            send_notification("Keylane", answer[:300])
            maybe_speak_notify(answer[:500])
    except Exception:  # noqa: BLE001
        logger.exception("scheduled task failed")


def schedule_cron(task_id: str, cron: str, prompt: str) -> str:
    sched = get_scheduler()
    parts = cron.split()
    if len(parts) != 5:
        raise ValueError("cron must be 5 fields: min hour dom month dow")
    trigger = CronTrigger(
        minute=parts[0],
        hour=parts[1],
        day=parts[2],
        month=parts[3],
        day_of_week=parts[4],
    )
    sched.add_job(_run_agent_prompt, trigger, args=[prompt], id=task_id, replace_existing=True)
    store = get_store()
    with store._connect() as conn:  # noqa: SLF001
        conn.execute(
            "INSERT OR REPLACE INTO scheduled_tasks (id, kind, schedule, prompt, enabled, created_at) VALUES (?,?,?,?,1,?)",
            (task_id, "cron", cron, prompt, _utcnow()),
        )
    return task_id


def schedule_at(task_id: str, run_at: str, prompt: str) -> str:
    sched = get_scheduler()
    when = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
    sched.add_job(
        _run_agent_prompt,
        DateTrigger(run_date=when),
        args=[prompt],
        id=task_id,
        replace_existing=True,
    )
    return task_id


def run_background(prompt: str) -> str:
    job_id = str(uuid.uuid4())

    def _worker() -> None:
        store = get_store()
        with store._connect() as conn:  # noqa: SLF001
            conn.execute(
                "INSERT INTO background_jobs (id, prompt, status, created_at) VALUES (?,?,?,?)",
                (job_id, prompt, "running", _utcnow()),
            )
        try:
            _run_agent_prompt(prompt, notify=True)
            status, result = "done", "completed"
        except Exception as exc:  # noqa: BLE001
            status, result = "error", str(exc)
        with store._connect() as conn:  # noqa: SLF001
            conn.execute(
                "UPDATE background_jobs SET status=?, result=?, finished_at=? WHERE id=?",
                (status, result, _utcnow(), job_id),
            )

    threading.Thread(target=_worker, daemon=True).start()
    return job_id
