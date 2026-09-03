"""The once-a-day look for a newer Keylane.

It writes a note to the inbox and stops. An assistant that replaces its own
running code while its owner is asleep is not a feature, so nothing here
installs anything — the note carries the version and the release notes, and
applying it is a button in Settings.

The job is scheduled in memory rather than persisted with the reminders,
because it is derived from a setting: turning the setting off should stop it,
not leave a stale row in the task store to be replayed on the next start.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

JOB_ID = "keylane:update-check"

# Not midnight. Every scheduled thing in the world fires at midnight, and a
# check that lands mid-morning is one the user is awake to read.
CHECK_HOUR = 10
CHECK_MINUTE = 17


def check_and_notify() -> None:
    """Look once, and file a note only when there is something to say."""
    from memory.store import push_inbox
    from updater.github import check_for_update

    from daemon.config import get_section

    channel = str(get_section("updates").get("channel", "stable"))
    try:
        status = check_for_update(channel, force=True)
    except Exception:  # noqa: BLE001
        logger.debug("daily update check failed", exc_info=True)
        return

    if not status.available:
        logger.info("update check: %s is current", status.current)
        return

    notes = (status.notes or "").strip()
    body = f"Keylane {status.latest_version} is available (you have {status.current})."
    if notes:
        first = "\n".join(notes.splitlines()[:6])
        body = f"{body}\n\n{first}"
    body += "\n\nInstall it from Settings → About."

    push_inbox(
        f"Keylane {status.latest_version} is available",
        body,
        kind="note",
        source="update-check",
    )
    logger.info("update available: %s", status.latest_version)


def install(enabled: bool | None = None) -> bool:
    """Arm or disarm the daily check to match settings. Returns whether it is on."""
    from apscheduler.triggers.cron import CronTrigger

    from daemon.config import get_section
    from scheduler.jobs import get_scheduler

    if enabled is None:
        enabled = bool(get_section("updates").get("check_daily", True))

    scheduler = get_scheduler()
    existing = scheduler.get_job(JOB_ID)
    if not enabled:
        if existing is not None:
            scheduler.remove_job(JOB_ID)
        return False

    scheduler.add_job(
        check_and_notify,
        CronTrigger(hour=CHECK_HOUR, minute=CHECK_MINUTE),
        id=JOB_ID,
        replace_existing=True,
    )
    return True
