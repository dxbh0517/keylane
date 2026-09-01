"""Background jobs, reminders, and scheduled watchers.

Every job is written to SQLite before it is handed to APScheduler, and
``restore_scheduled_tasks`` replays them at daemon start. Without that, a
reminder set on Monday quietly disappears the next time the daemon restarts —
which is the one failure mode a reminder must not have.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from memory.store import (
    disable_scheduled_task,
    get_store,
    list_scheduled_tasks,
    mark_task_run,
    push_inbox,
    upsert_scheduled_task,
)
from scheduler.timeparse import describe, local_now, parse_when, to_utc_iso

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()

# A one-shot that we missed by less than this still fires (late) on restart.
_MISSED_GRACE = timedelta(hours=12)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is None:
        with _lock:
            if _scheduler is None:
                _scheduler = BackgroundScheduler()
                _scheduler.start()
    return _scheduler


def _deliver(title: str, body: str, *, source: str, speak: bool = True) -> None:
    """Desktop notification + inbox row, so nothing is lost if it is missed."""
    from notify.desktop import send_notification

    push_inbox(title, body, kind="result", source=source)
    send_notification(title, body[:300])
    if speak:
        from notify.tts_gate import maybe_speak_notify

        maybe_speak_notify(body[:500])


def deliver_reminder(task_id: str, text: str, *, late: bool = False) -> None:
    """Fire a plain reminder — no model call, so it works even on a cold NPU."""
    body = f"(missed while offline) {text}" if late else text
    _deliver("Keylane reminder", body, source=f"reminder:{task_id}")
    mark_task_run(task_id)
    disable_scheduled_task(task_id)


def run_agent_prompt(prompt: str, *, task_id: str = "", notify: bool = True) -> str:
    """Run one agent turn headlessly and deliver the answer."""
    import asyncio

    from agent.loop import AIAgent

    async def _inner() -> str:
        # Nobody is watching the clock on scheduled work, so it takes the
        # background route and the larger model when one is configured.
        agent = AIAgent(route="background")
        result = await agent.run(prompt)
        return result.answer

    try:
        answer = asyncio.run(_inner())
    except Exception as exc:  # noqa: BLE001
        logger.exception("scheduled task failed")
        if notify:
            _deliver("Keylane task failed", str(exc)[:300], source=f"task:{task_id}", speak=False)
        return ""

    if task_id:
        mark_task_run(task_id)
    if notify and answer.strip():
        _deliver("Keylane", answer, source=f"task:{task_id}" if task_id else "background")
    return answer


# Kept for callers that predate the rename.
_run_agent_prompt = run_agent_prompt


def _cron_trigger(cron: str) -> CronTrigger:
    parts = cron.split()
    if len(parts) != 5:
        raise ValueError("cron must be 5 fields: minute hour day month day_of_week")
    return CronTrigger(
        minute=parts[0],
        hour=parts[1],
        day=parts[2],
        month=parts[3],
        day_of_week=parts[4],
    )


def schedule_cron(task_id: str, cron: str, prompt: str, *, title: str = "", persist: bool = True) -> str:
    trigger = _cron_trigger(cron)
    get_scheduler().add_job(
        run_agent_prompt,
        trigger,
        kwargs={"prompt": prompt, "task_id": task_id},
        id=task_id,
        replace_existing=True,
    )
    if persist:
        upsert_scheduled_task(task_id, "cron", prompt, schedule=cron, title=title)
    return task_id


def schedule_at(
    task_id: str,
    run_at: str | datetime,
    prompt: str,
    *,
    kind: str = "at",
    title: str = "",
    persist: bool = True,
) -> str:
    when = run_at if isinstance(run_at, datetime) else datetime.fromisoformat(str(run_at).replace("Z", "+00:00"))
    if when.tzinfo is None:
        when = when.astimezone()

    if kind == "reminder":
        job, kwargs = deliver_reminder, {"task_id": task_id, "text": prompt}
    else:
        job, kwargs = run_agent_prompt, {"prompt": prompt, "task_id": task_id}

    get_scheduler().add_job(job, DateTrigger(run_date=when), kwargs=kwargs, id=task_id, replace_existing=True)
    if persist:
        upsert_scheduled_task(task_id, kind, prompt, run_at=to_utc_iso(when), title=title)
    return task_id


def create_reminder(text: str, when: str) -> dict[str, Any]:
    """Schedule a reminder from a natural-language time such as 'tomorrow 9am'."""
    target = parse_when(when)
    if target is None:
        return {
            "error": f"Could not understand the time {when!r}.",
            "hint": "Try 'in 30 minutes', 'tomorrow at 9am', 'Friday 17:00', or an ISO timestamp.",
        }
    if target <= local_now():
        return {"error": f"{describe(target)} is in the past."}
    task_id = f"remind-{uuid.uuid4().hex[:8]}"
    schedule_at(task_id, target, text, kind="reminder", title=text[:60])
    return {"id": task_id, "text": text, "when": to_utc_iso(target), "when_human": describe(target)}


def create_watcher(name: str, prompt: str, cron: str) -> dict[str, Any]:
    """A recurring agent task, e.g. a morning sweep of calendar and mail."""
    task_id = f"watch-{name.strip().lower().replace(' ', '-')[:24] or uuid.uuid4().hex[:8]}"
    try:
        schedule_cron(task_id, cron, prompt, title=name)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"id": task_id, "name": name, "cron": cron}


def cancel_task(task_id: str) -> dict[str, Any]:
    try:
        get_scheduler().remove_job(task_id)
    except Exception:  # noqa: BLE001 — already fired or never registered
        pass
    existed = disable_scheduled_task(task_id)
    return {"cancelled": task_id} if existed else {"error": f"no task with id {task_id}"}


def list_tasks() -> list[dict[str, Any]]:
    rows = list_scheduled_tasks()
    for row in rows:
        if row.get("run_at"):
            try:
                row["when_human"] = describe(datetime.fromisoformat(row["run_at"]))
            except ValueError:
                pass
    return rows


def restore_scheduled_tasks() -> int:
    """Re-register persisted tasks at startup; fire recently-missed one-shots."""
    restored = 0
    for row in list_scheduled_tasks():
        task_id = row["id"]
        prompt = row.get("prompt") or ""
        kind = row.get("kind") or "at"
        try:
            if kind == "cron" and row.get("schedule"):
                schedule_cron(task_id, row["schedule"], prompt, title=row.get("title") or "", persist=False)
                restored += 1
                continue

            run_at = row.get("run_at")
            if not run_at:
                continue
            when = datetime.fromisoformat(run_at)
            now = datetime.now(timezone.utc)
            if when > now:
                schedule_at(task_id, when, prompt, kind=kind, title=row.get("title") or "", persist=False)
                restored += 1
            elif now - when <= _MISSED_GRACE and not row.get("last_run"):
                logger.info("firing missed task %s (due %s)", task_id, run_at)
                if kind == "reminder":
                    deliver_reminder(task_id, prompt, late=True)
                else:
                    threading.Thread(
                        target=run_agent_prompt,
                        kwargs={"prompt": prompt, "task_id": task_id},
                        daemon=True,
                    ).start()
            else:
                disable_scheduled_task(task_id)
        except Exception:  # noqa: BLE001 — one bad row must not block the rest
            logger.exception("could not restore task %s", task_id)
    logger.info("restored %s scheduled task(s)", restored)
    return restored


def run_background(prompt: str) -> str:
    """Start one agent run in the background and return its job id.

    Goes through the job registry rather than a bare thread, so the run has an
    id the model can read from and stop — and so the depth cap applies, which
    is what stops a background agent from starting another one forever.
    """
    from seams import get_context
    from seams.jobs import Job

    store = get_store()

    def _work(job: Job) -> str:
        with store._connect() as conn:  # noqa: SLF001
            conn.execute(
                "INSERT INTO background_jobs (id, prompt, status, created_at) VALUES (?,?,?,?)",
                (job.id, prompt, "running", _utcnow()),
            )
        try:
            answer = run_agent_prompt(prompt, task_id=job.id, notify=not job.cancel.is_set())
            status, result = "done", answer[:4000]
        except Exception:
            with store._connect() as conn:  # noqa: SLF001
                conn.execute(
                    "UPDATE background_jobs SET status=?, finished_at=? WHERE id=?",
                    ("error", _utcnow(), job.id),
                )
            raise
        with store._connect() as conn:  # noqa: SLF001
            conn.execute(
                "UPDATE background_jobs SET status=?, result=?, finished_at=? WHERE id=?",
                (status, result, _utcnow(), job.id),
            )
        return result

    return get_context().jobs.start(kind="agent", label=prompt[:80], work=_work).id


def background_jobs(limit: int = 20) -> list[dict[str, Any]]:
    with get_store()._connect() as conn:  # noqa: SLF001
        rows = conn.execute(
            "SELECT id, prompt, status, result, created_at, finished_at"
            " FROM background_jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
