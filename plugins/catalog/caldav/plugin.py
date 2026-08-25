"""Minimal CalDAV tools using caldav + icalendar when installed."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.plugins.base import (
    BasePlugin,
    PluginHealth,
    PluginKind,
    SettingField,
    SettingType,
)
from app.tools.base import (
    BaseTool,
    ToolDanger,
    ToolResult,
    int_prop,
    object_schema,
    string_prop,
)

logger = logging.getLogger(__name__)


def _client(url: str, username: str, password: str):
    import caldav

    return caldav.DAVClient(url=url, username=username or None, password=password or None)


class ListEventsTool(BaseTool):
    name = "caldav.list_events"
    description = "List upcoming events from the configured CalDAV account."
    danger = ToolDanger.SAFE
    category = "calendar"

    def __init__(self, plugin: "Plugin") -> None:
        self._plugin = plugin

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "days": int_prop("Days ahead.", default=7),
                "limit": int_prop("Max events.", default=20),
            }
        )

    def availability(self) -> str | None:
        if not self._plugin.settings.get("url"):
            return "CalDAV URL is not configured"
        try:
            import caldav  # noqa: F401
        except ImportError:
            return "Python package 'caldav' is not installed"
        return None

    async def run(self, args: dict[str, Any]) -> ToolResult:
        days = max(1, int(args.get("days") or 7))
        limit = max(1, min(int(args.get("limit") or 20), 50))
        settings = self._plugin.settings
        try:
            client = _client(
                str(settings.get("url") or ""),
                str(settings.get("username") or ""),
                str(settings.get("password") or ""),
            )
            principal = client.principal()
            calendars = principal.calendars()
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"CalDAV connection failed: {exc}")
        if not calendars:
            return ToolResult.failure("No CalDAV calendars found.")

        start = datetime.now(timezone.utc)
        end = start + timedelta(days=days)
        events: list[dict[str, Any]] = []
        for calendar in calendars:
            try:
                results = calendar.search(start=start, end=end, event=True, expand=True)
            except Exception as exc:  # noqa: BLE001
                logger.debug("caldav search failed: %s", exc)
                continue
            for item in results:
                try:
                    vevent = item.vobject_instance.vevent
                    summary = str(getattr(vevent, "summary", None).value if hasattr(vevent, "summary") else "")
                    dtstart = str(getattr(vevent, "dtstart", None).value if hasattr(vevent, "dtstart") else "")
                except Exception:  # noqa: BLE001
                    summary, dtstart = "", ""
                events.append(
                    {
                        "calendar": str(getattr(calendar, "name", "") or "Calendar"),
                        "summary": summary,
                        "start": dtstart,
                    }
                )
        events = events[:limit]
        if not events:
            return ToolResult.success(f"No events in the next {days} day(s).", data={"events": []})
        lines = [f"- {e['start']}  {e['summary']}" for e in events]
        return ToolResult.success("\n".join(lines), data={"events": events})


class CreateEventTool(BaseTool):
    name = "caldav.create_event"
    description = "Create a CalDAV event. Confirm with the user first."
    danger = ToolDanger.SENSITIVE
    category = "calendar"

    def __init__(self, plugin: "Plugin") -> None:
        self._plugin = plugin

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "summary": string_prop("Event title."),
                "start": string_prop("ISO-8601 start."),
                "end": string_prop("ISO-8601 end (optional)."),
            },
            required=["summary", "start"],
        )

    def availability(self) -> str | None:
        return ListEventsTool(self._plugin).availability()

    async def run(self, args: dict[str, Any]) -> ToolResult:
        from icalendar import Calendar, Event

        summary = str(args.get("summary") or "").strip()
        start_raw = str(args.get("start") or "").strip()
        if not summary or not start_raw:
            return ToolResult.failure("summary and start are required.")
        start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
        end_raw = str(args.get("end") or "").strip()
        end = (
            datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
            if end_raw
            else start + timedelta(hours=1)
        )
        settings = self._plugin.settings
        try:
            client = _client(
                str(settings.get("url") or ""),
                str(settings.get("username") or ""),
                str(settings.get("password") or ""),
            )
            calendars = client.principal().calendars()
            if not calendars:
                return ToolResult.failure("No CalDAV calendars found.")
            cal = Calendar()
            cal.add("prodid", "-//Keylane//")
            cal.add("version", "2.0")
            event = Event()
            event.add("summary", summary)
            event.add("dtstart", start)
            event.add("dtend", end)
            cal.add_component(event)
            calendars[0].save_event(cal.to_ical())
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Could not create event: {exc}")
        return ToolResult.success(f"Created '{summary}' at {start.isoformat()}.")


class Plugin(BasePlugin):
    id = "caldav"
    name = "CalDAV Calendar"
    kind = PluginKind.NATIVE
    description = "Remote CalDAV calendars."
    removable = True
    worker_id = None
    cloud = False

    def default_settings(self) -> dict[str, Any]:
        return {
            "url": "",
            "username": "",
            "password": "",
            **super().default_settings(),
        }

    def settings_schema(self) -> list[SettingField]:
        return [
            SettingField(key="url", label="CalDAV URL", type=SettingType.STRING, required=True),
            SettingField(key="username", label="Username", type=SettingType.STRING),
            SettingField(key="password", label="Password", type=SettingType.SECRET),
        ]

    def tools(self) -> list[BaseTool]:
        tools = [ListEventsTool(self), CreateEventTool(self)]
        for tool in tools:
            tool.source = self.id
        return tools

    async def health(self) -> PluginHealth:
        reason = ListEventsTool(self).availability()
        if reason:
            return PluginHealth(ok=False, detail=reason)
        try:
            client = _client(
                str(self.settings.get("url") or ""),
                str(self.settings.get("username") or ""),
                str(self.settings.get("password") or ""),
            )
            calendars = client.principal().calendars()
            return PluginHealth(
                ok=True, detail=f"{len(calendars)} calendar(s) reachable"
            )
        except Exception as exc:  # noqa: BLE001
            return PluginHealth(ok=False, detail=str(exc))


def create_plugin(settings: dict[str, Any] | None = None) -> Plugin:
    return Plugin(settings or {})
