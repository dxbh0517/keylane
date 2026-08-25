"""GNOME Calendar tools via Evolution Data Server (ECal)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.plugins.base import BasePlugin, PluginHealth, PluginKind
from app.tools.base import (
    BaseTool,
    ToolDanger,
    ToolResult,
    int_prop,
    object_schema,
    string_prop,
)

logger = logging.getLogger(__name__)


def _eds_available() -> str | None:
    try:
        import gi

        gi.require_version("EDataServer", "1.2")
        gi.require_version("ECal", "2.0")
        from gi.repository import ECal, EDataServer  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return (
            "Evolution Data Server Python bindings unavailable "
            f"({exc}). Install evolution-data-server and the ECal GIR typelibs."
        )
    return None


def _list_clients():
    from gi.repository import ECal, EDataServer

    registry = EDataServer.SourceRegistry.new_sync(None)
    sources = EDataServer.SourceRegistry.list_sources(
        registry, EDataServer.SOURCE_EXTENSION_CALENDAR
    )
    clients = []
    for source in sources:
        try:
            client = ECal.Client.connect_sync(
                source, ECal.ClientSourceType.EVENTS, 2, None
            )
            if client is not None:
                clients.append((source.get_display_name() or "Calendar", client))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not open calendar source: %s", exc)
    return clients


def _event_dict(comp, calendar_name: str) -> dict[str, Any]:
    summary = ""
    try:
        summaries = comp.get_summaries()
        if summaries:
            summary = summaries[0].get_value() or ""
    except Exception:  # noqa: BLE001
        pass
    start = ""
    end = ""
    try:
        dtstart = comp.get_dtstart()
        if dtstart and dtstart.get_value():
            start = datetime.fromtimestamp(
                dtstart.get_value().as_timet(), tz=timezone.utc
            ).isoformat()
    except Exception:  # noqa: BLE001
        pass
    try:
        dtend = comp.get_dtend()
        if dtend and dtend.get_value():
            end = datetime.fromtimestamp(
                dtend.get_value().as_timet(), tz=timezone.utc
            ).isoformat()
    except Exception:  # noqa: BLE001
        pass
    uid = ""
    try:
        uid = comp.get_uid() or ""
    except Exception:  # noqa: BLE001
        pass
    return {
        "calendar": calendar_name,
        "uid": uid,
        "summary": summary,
        "start": start,
        "end": end,
    }


class ListUpcomingTool(BaseTool):
    name = "list_upcoming"
    description = (
        "List upcoming events from the Fedora / GNOME Calendar (Evolution Data Server)."
    )
    danger = ToolDanger.SAFE
    category = "calendar"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "days": int_prop("How many days ahead to look.", default=7),
                "limit": int_prop("Max events to return.", default=20),
            }
        )

    def availability(self) -> str | None:
        return _eds_available()

    async def run(self, args: dict[str, Any]) -> ToolResult:
        days = max(1, int(args.get("days") or 7))
        limit = max(1, min(int(args.get("limit") or 20), 50))
        try:
            clients = _list_clients()
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Could not open calendars: {exc}")
        if not clients:
            return ToolResult.failure("No GNOME / Evolution calendars found.")

        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)
        # EDS sexp ranges are exclusive; pad by a second.
        start_ts = int(now.timestamp()) - 1
        end_ts = int(end.timestamp()) + 1
        sexp = f"(occur-in-time-range? (make-time {start_ts}) (make-time {end_ts}))"

        events: list[dict[str, Any]] = []
        for name, client in clients:
            try:
                ok, comps = client.get_object_list_as_comps_sync(sexp, None)
            except Exception as exc:  # noqa: BLE001
                logger.debug("list events failed on %s: %s", name, exc)
                continue
            if not ok or not comps:
                continue
            for comp in comps:
                events.append(_event_dict(comp, name))

        events.sort(key=lambda e: e.get("start") or "")
        events = events[:limit]
        if not events:
            return ToolResult.success(
                f"No events in the next {days} day(s).", data={"events": []}
            )
        lines = [
            f"- {e['start'][:16]}  {e['summary'] or '(no title)'}  [{e['calendar']}]"
            for e in events
        ]
        return ToolResult.success("\n".join(lines), data={"events": events})


class CreateEventTool(BaseTool):
    name = "create_event"
    description = (
        "Create a calendar event in GNOME Calendar / Evolution Data Server. "
        "Always confirm with the user before calling."
    )
    danger = ToolDanger.SENSITIVE
    category = "calendar"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "summary": string_prop("Event title."),
                "start": string_prop("ISO-8601 start time (UTC or with offset)."),
                "end": string_prop("ISO-8601 end time. Defaults to one hour after start."),
                "description": string_prop("Optional notes."),
                "calendar": string_prop("Optional calendar display name."),
            },
            required=["summary", "start"],
        )

    def availability(self) -> str | None:
        return _eds_available()

    async def run(self, args: dict[str, Any]) -> ToolResult:
        from gi.repository import ECal

        summary = str(args.get("summary") or "").strip()
        start_raw = str(args.get("start") or "").strip()
        if not summary or not start_raw:
            return ToolResult.failure("summary and start are required.")
        try:
            start = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
        except ValueError:
            return ToolResult.failure("Could not parse start time.")
        end_raw = str(args.get("end") or "").strip()
        if end_raw:
            try:
                end = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
            except ValueError:
                return ToolResult.failure("Could not parse end time.")
        else:
            end = start + timedelta(hours=1)

        try:
            clients = _list_clients()
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Could not open calendars: {exc}")
        if not clients:
            return ToolResult.failure("No calendars available.")

        wanted = str(args.get("calendar") or "").strip().lower()
        name, client = clients[0]
        if wanted:
            for candidate_name, candidate in clients:
                if wanted in candidate_name.lower():
                    name, client = candidate_name, candidate
                    break

        ical = (
            "BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\n"
            f"SUMMARY:{summary}\r\n"
            f"DTSTART:{start.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}\r\n"
            f"DTEND:{end.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}\r\n"
        )
        description = str(args.get("description") or "").strip()
        if description:
            ical += f"DESCRIPTION:{description}\r\n"
        ical += "END:VEVENT\r\nEND:VCALENDAR\r\n"

        try:
            comp = ECal.Component.new_from_string(ical)
            # Some bindings expect the VEVENT only.
            if comp is None:
                from gi.repository import ICalGLib

                icalcomp = ICalGLib.Component.new_from_string(ical)
                uid = client.create_object_sync(icalcomp, None)
            else:
                uid = client.create_object_sync(comp.get_icalcomponent(), None)
        except Exception as exc:  # noqa: BLE001
            logger.exception("create_event failed")
            return ToolResult.failure(f"Could not create event: {exc}")

        return ToolResult.success(
            f"Created '{summary}' on {name} at {start.isoformat()}.",
            data={"uid": uid, "calendar": name, "summary": summary, "start": start.isoformat()},
        )


class Plugin(BasePlugin):
    id = "gnome-calendar"
    name = "GNOME Calendar"
    kind = PluginKind.NATIVE
    description = "Fedora / GNOME Calendar via Evolution Data Server."
    removable = True
    worker_id = None
    cloud = False

    def tools(self) -> list[BaseTool]:
        # Namespace under the plugin id so the assistant sees gnome-calendar.*.
        tools = [ListUpcomingTool(), CreateEventTool()]
        for tool in tools:
            tool.name = f"gnome-calendar.{tool.name.split('.')[-1]}"
            tool.source = self.id
        return tools

    async def health(self) -> PluginHealth:
        reason = _eds_available()
        if reason:
            return PluginHealth(ok=False, detail=reason)
        try:
            clients = _list_clients()
        except Exception as exc:  # noqa: BLE001
            return PluginHealth(ok=False, detail=str(exc))
        if not clients:
            return PluginHealth(ok=False, detail="No calendar sources found")
        names = ", ".join(name for name, _ in clients[:5])
        return PluginHealth(ok=True, detail=f"Calendars: {names}")


def create_plugin(settings: dict[str, Any] | None = None) -> Plugin:
    return Plugin(settings or {})
