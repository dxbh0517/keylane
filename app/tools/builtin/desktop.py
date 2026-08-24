"""Desktop tools — launch applications, open URLs, notify, clipboard, media."""

from __future__ import annotations

import asyncio
import configparser
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from app.tools.base import (
    BaseTool,
    ToolDanger,
    ToolResult,
    int_prop,
    object_schema,
    string_prop,
)

logger = logging.getLogger(__name__)

DESKTOP_DIRS = [
    Path.home() / ".local/share/applications",
    Path("/usr/share/applications"),
    Path("/usr/local/share/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path.home() / ".local/share/flatpak/exports/share/applications",
    Path("/var/lib/snapd/desktop/applications"),
]

# Field codes a .desktop Exec line may carry; none of them apply to us.
_FIELD_CODES = re.compile(r"%[fFuUdDnNickvm]")


async def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    return (
        proc.returncode or 0,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


async def _spawn(cmd: list[str], cwd: str | None = None) -> None:
    """Start a detached GUI process; we do not wait for it."""
    await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )


class DesktopEntry:
    __slots__ = ("path", "app_id", "name", "generic_name", "comment", "exec_line", "no_display")

    def __init__(self, path: Path, data: dict[str, str]) -> None:
        self.path = path
        self.app_id = path.stem
        self.name = data.get("Name", path.stem)
        self.generic_name = data.get("GenericName", "")
        self.comment = data.get("Comment", "")
        self.exec_line = data.get("Exec", "")
        self.no_display = data.get("NoDisplay", "false").strip().lower() == "true"

    def haystack(self) -> str:
        return " ".join(
            (self.name, self.generic_name, self.comment, self.app_id)
        ).lower()


def _parse_desktop_file(path: Path) -> DesktopEntry | None:
    parser = configparser.RawConfigParser(strict=False, interpolation=None)
    try:
        parser.read(path, encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None
    section = "Desktop Entry"
    if not parser.has_section(section):
        return None
    data = {key: parser.get(section, key) for key in parser.options(section)}
    # RawConfigParser lowercases keys; restore the ones we care about.
    normalized = {
        "Name": data.get("name", path.stem),
        "GenericName": data.get("genericname", ""),
        "Comment": data.get("comment", ""),
        "Exec": data.get("exec", ""),
        "NoDisplay": data.get("nodisplay", "false"),
        "Type": data.get("type", "Application"),
    }
    if normalized["Type"].strip() != "Application":
        return None
    return DesktopEntry(path, normalized)


def scan_desktop_entries() -> list[DesktopEntry]:
    seen: dict[str, DesktopEntry] = {}
    for directory in DESKTOP_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.desktop")):
            if path.stem in seen:
                continue  # earlier directories win (user overrides system)
            entry = _parse_desktop_file(path)
            if entry is not None and not entry.no_display:
                seen[path.stem] = entry
    return list(seen.values())


# Words people add that are not part of any application's name.
_FILLER = {"app", "application", "the", "a", "an", "program", "gnome", "kde", "please"}


def find_application(query: str) -> DesktopEntry | None:
    """Best-effort match for a spoken or typed application name.

    People say "the gnome text editor" for an entry called "Text Editor", so
    matching is token-based rather than a plain substring test: every meaningful
    word in the query must appear somewhere in the entry, and closer matches
    (exact name, prefix, whole phrase) still win.
    """
    query = (query or "").strip().lower()
    if not query:
        return None
    tokens = [t for t in re.split(r"[\s_.-]+", query) if t]
    meaningful = [t for t in tokens if t not in _FILLER] or tokens
    entries = scan_desktop_entries()

    def score(entry: DesktopEntry) -> int:
        name = entry.name.lower()
        app_id = entry.app_id.lower()
        haystack = entry.haystack()
        if query in (name, app_id):
            return 0
        if name.startswith(query) or app_id.startswith(query):
            return 1
        if query in name:
            return 2
        if query in haystack:
            return 3
        # Every meaningful word present, in the visible name.
        if all(word in name for word in meaningful):
            return 4
        # Every meaningful word present anywhere (id, comment, generic name).
        if all(word in haystack for word in meaningful):
            return 5
        return 99

    ranked = sorted(
        ((score(e), len(e.name), e.name.lower(), e) for e in entries),
        key=lambda item: (item[0], item[1], item[2]),
    )
    if ranked and ranked[0][0] < 99:
        return ranked[0][3]
    return None


class ListApplicationsTool(BaseTool):
    name = "list_applications"
    description = (
        "List installed desktop applications, optionally filtered by a search term. "
        "Use this to confirm an app exists before opening it."
    )
    danger = ToolDanger.SAFE
    category = "desktop"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "query": string_prop("Optional substring to filter application names."),
                "limit": int_prop("Maximum entries to return (default 25).", default=25),
            }
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip().lower()
        limit = max(1, min(int(args.get("limit") or 25), 200))
        entries = scan_desktop_entries()
        if query:
            entries = [e for e in entries if query in e.haystack()]
        entries.sort(key=lambda e: e.name.lower())
        shown = entries[:limit]
        listing = "\n".join(f"{e.name} ({e.app_id})" for e in shown)
        return ToolResult.success(
            listing or "No matching applications installed.",
            data={
                "count": len(entries),
                "applications": [
                    {"name": e.name, "id": e.app_id, "comment": e.comment} for e in shown
                ],
            },
        )


class OpenApplicationTool(BaseTool):
    name = "open_application"
    description = (
        "Launch an installed desktop application by name, for example 'Firefox', "
        "'files' or 'code'. Matches against installed .desktop entries."
    )
    danger = ToolDanger.SENSITIVE
    category = "desktop"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "application": string_prop("Application name or desktop id to launch."),
            },
            required=["application"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("application") or "").strip()
        if not query:
            return ToolResult.failure("No application name given.")
        entry = find_application(query)
        if entry is None:
            return ToolResult.failure(
                f"No installed application matches '{query}'. "
                "Call list_applications to see what is available."
            )

        launcher = shutil.which("gtk-launch")
        if launcher:
            try:
                await _spawn([launcher, entry.path.stem])
                return ToolResult.success(
                    f"Launched {entry.name}.",
                    data={"application": entry.name, "desktop_id": entry.app_id},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("gtk-launch failed for %s: %s", entry.app_id, exc)

        gio = shutil.which("gio")
        if gio:
            try:
                await _spawn([gio, "launch", str(entry.path)])
                return ToolResult.success(
                    f"Launched {entry.name}.",
                    data={"application": entry.name, "desktop_id": entry.app_id},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("gio launch failed for %s: %s", entry.app_id, exc)

        # Last resort: run the Exec line with field codes stripped.
        command = _FIELD_CODES.sub("", entry.exec_line).strip()
        if not command:
            return ToolResult.failure(f"{entry.name} has no runnable Exec line.")
        parts = command.split()
        try:
            await _spawn(parts)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Could not launch {entry.name}: {exc}")
        return ToolResult.success(
            f"Launched {entry.name}.",
            data={"application": entry.name, "desktop_id": entry.app_id},
        )


class OpenUrlTool(BaseTool):
    name = "open_url"
    description = (
        "Open a URL, file path or folder in the user's default application "
        "(browser, file manager, viewer)."
    )
    danger = ToolDanger.SENSITIVE
    category = "desktop"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {"target": string_prop("URL such as https://… or an absolute file path.")},
            required=["target"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        target = str(args.get("target") or "").strip()
        if not target:
            return ToolResult.failure("No target given.")
        if "://" not in target:
            path = Path(target).expanduser()
            if not path.exists():
                return ToolResult.failure(f"Path does not exist: {path}")
            target = str(path)
        opener = shutil.which("xdg-open") or shutil.which("gio")
        if opener is None:
            return ToolResult.failure("Neither xdg-open nor gio is installed.")
        cmd = [opener, target] if opener.endswith("xdg-open") else [opener, "open", target]
        try:
            await _spawn(cmd)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Could not open {target}: {exc}")
        return ToolResult.success(f"Opened {target}.", data={"target": target})

    def availability(self) -> str | None:
        if shutil.which("xdg-open") or shutil.which("gio"):
            return None
        return "xdg-open / gio not installed"


class NotifyTool(BaseTool):
    name = "send_notification"
    description = "Show a desktop notification to the user."
    danger = ToolDanger.SAFE
    category = "desktop"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "title": string_prop("Notification headline."),
                "body": string_prop("Notification body text."),
                "urgency": string_prop(
                    "low, normal or critical.",
                    enum=["low", "normal", "critical"],
                    default="normal",
                ),
            },
            required=["title"],
        )

    def availability(self) -> str | None:
        return None if shutil.which("notify-send") else "notify-send not installed"

    async def run(self, args: dict[str, Any]) -> ToolResult:
        binary = shutil.which("notify-send")
        if binary is None:
            return ToolResult.failure("notify-send is not installed.")
        title = str(args.get("title") or "Keylane")
        body = str(args.get("body") or "")
        urgency = str(args.get("urgency") or "normal")
        if urgency not in {"low", "normal", "critical"}:
            urgency = "normal"
        cmd = [binary, "-a", "Keylane", "-u", urgency, "-i", "keylane", title]
        if body:
            cmd.append(body)
        code, _out, err = await _run(cmd, timeout=10.0)
        if code != 0:
            return ToolResult.failure(err.strip() or "notify-send failed")
        return ToolResult.success("Notification shown.")


class ClipboardReadTool(BaseTool):
    name = "read_clipboard"
    description = "Read the current text contents of the system clipboard."
    danger = ToolDanger.SAFE
    category = "desktop"

    def _binary(self) -> list[str] | None:
        if shutil.which("wl-paste"):
            return [shutil.which("wl-paste"), "--no-newline"]  # type: ignore[list-item]
        if shutil.which("xclip"):
            return [shutil.which("xclip"), "-selection", "clipboard", "-o"]  # type: ignore[list-item]
        return None

    def availability(self) -> str | None:
        return None if self._binary() else "wl-paste / xclip not installed"

    async def run(self, args: dict[str, Any]) -> ToolResult:
        cmd = self._binary()
        if cmd is None:
            return ToolResult.failure("No clipboard tool (wl-paste or xclip) installed.")
        try:
            code, out, err = await _run(cmd, timeout=10.0)
        except asyncio.TimeoutError:
            return ToolResult.failure("Clipboard read timed out.")
        if code != 0:
            return ToolResult.failure(err.strip() or "Clipboard read failed")
        return ToolResult.success(out, data={"length": len(out)})


class ClipboardWriteTool(BaseTool):
    name = "write_clipboard"
    description = "Replace the system clipboard contents with the given text."
    danger = ToolDanger.SENSITIVE
    category = "desktop"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {"text": string_prop("Text to place on the clipboard.")},
            required=["text"],
        )

    def _binary(self) -> list[str] | None:
        if shutil.which("wl-copy"):
            return [shutil.which("wl-copy")]  # type: ignore[list-item]
        if shutil.which("xclip"):
            return [shutil.which("xclip"), "-selection", "clipboard"]  # type: ignore[list-item]
        return None

    def availability(self) -> str | None:
        return None if self._binary() else "wl-copy / xclip not installed"

    async def run(self, args: dict[str, Any]) -> ToolResult:
        cmd = self._binary()
        if cmd is None:
            return ToolResult.failure("No clipboard tool (wl-copy or xclip) installed.")
        text = str(args.get("text") or "")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _out, err = await proc.communicate(text.encode("utf-8"))
        if (proc.returncode or 0) != 0:
            return ToolResult.failure(err.decode("utf-8", "replace").strip() or "Copy failed")
        return ToolResult.success(f"Copied {len(text)} characters to the clipboard.")


class MediaControlTool(BaseTool):
    name = "media_control"
    description = (
        "Control media playback (play, pause, play-pause, next, previous, stop) "
        "for the active player."
    )
    danger = ToolDanger.SENSITIVE
    category = "desktop"

    ACTIONS = ("play", "pause", "play-pause", "next", "previous", "stop", "status")

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {"action": string_prop("Playback action.", enum=list(self.ACTIONS))},
            required=["action"],
        )

    def availability(self) -> str | None:
        return None if shutil.which("playerctl") else "playerctl not installed"

    async def run(self, args: dict[str, Any]) -> ToolResult:
        binary = shutil.which("playerctl")
        if binary is None:
            return ToolResult.failure("playerctl is not installed.")
        action = str(args.get("action") or "").strip().lower()
        if action not in self.ACTIONS:
            return ToolResult.failure(f"action must be one of {', '.join(self.ACTIONS)}")
        code, out, err = await _run([binary, action], timeout=10.0)
        if code != 0:
            return ToolResult.failure(err.strip() or f"playerctl {action} failed")
        return ToolResult.success(out.strip() or f"Media {action} sent.")


class VolumeTool(BaseTool):
    name = "set_volume"
    description = "Set or mute the default audio output volume (0-100 percent)."
    danger = ToolDanger.SENSITIVE
    category = "desktop"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "percent": int_prop("Target volume 0-100.", minimum=0, maximum=100),
                "mute": string_prop(
                    "Set to 'on', 'off' or 'toggle' to change mute instead of volume.",
                    enum=["on", "off", "toggle"],
                ),
            }
        )

    def availability(self) -> str | None:
        if shutil.which("wpctl") or shutil.which("pactl"):
            return None
        return "wpctl / pactl not installed"

    async def run(self, args: dict[str, Any]) -> ToolResult:
        wpctl = shutil.which("wpctl")
        pactl = shutil.which("pactl")
        mute = str(args.get("mute") or "").strip().lower()

        if mute:
            flag = {"on": "1", "off": "0", "toggle": "toggle"}.get(mute)
            if flag is None:
                return ToolResult.failure("mute must be on, off or toggle")
            if wpctl:
                cmd = [wpctl, "set-mute", "@DEFAULT_AUDIO_SINK@", flag]
            elif pactl:
                cmd = [pactl, "set-sink-mute", "@DEFAULT_SINK@", flag]
            else:
                return ToolResult.failure("No PipeWire/PulseAudio control tool found.")
            code, _out, err = await _run(cmd, timeout=10.0)
            if code != 0:
                return ToolResult.failure(err.strip() or "Mute change failed")
            return ToolResult.success(f"Mute set to {mute}.")

        if args.get("percent") is None:
            return ToolResult.failure("Provide either percent or mute.")
        percent = max(0, min(int(args["percent"]), 100))
        if wpctl:
            cmd = [wpctl, "set-volume", "@DEFAULT_AUDIO_SINK@", f"{percent / 100:.2f}"]
        elif pactl:
            cmd = [pactl, "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"]
        else:
            return ToolResult.failure("No PipeWire/PulseAudio control tool found.")
        code, _out, err = await _run(cmd, timeout=10.0)
        if code != 0:
            return ToolResult.failure(err.strip() or "Volume change failed")
        return ToolResult.success(f"Volume set to {percent}%.")


class SystemInfoTool(BaseTool):
    name = "system_info"
    description = (
        "Report host facts: hostname, OS, uptime, CPU/memory/disk usage, "
        "battery and the current date and time."
    )
    danger = ToolDanger.SAFE
    category = "system"

    async def run(self, args: dict[str, Any]) -> ToolResult:
        import platform
        from datetime import datetime

        data: dict[str, Any] = {
            "hostname": platform.node(),
            "system": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "local_time": datetime.now().astimezone().isoformat(timespec="seconds"),
            "user": os.environ.get("USER") or os.environ.get("LOGNAME") or "",
        }

        try:
            os_release = Path("/etc/os-release").read_text(encoding="utf-8")
            match = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', os_release, re.MULTILINE)
            if match:
                data["distribution"] = match.group(1)
        except Exception:  # noqa: BLE001
            pass

        try:
            uptime_seconds = float(Path("/proc/uptime").read_text().split()[0])
            hours, remainder = divmod(int(uptime_seconds), 3600)
            data["uptime"] = f"{hours}h {remainder // 60}m"
        except Exception:  # noqa: BLE001
            pass

        try:
            meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
            total = int(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1))  # type: ignore[union-attr]
            available = int(re.search(r"MemAvailable:\s+(\d+)", meminfo).group(1))  # type: ignore[union-attr]
            data["memory_total_gb"] = round(total / 1024 / 1024, 1)
            data["memory_used_gb"] = round((total - available) / 1024 / 1024, 1)
        except Exception:  # noqa: BLE001
            pass

        try:
            usage = shutil.disk_usage(Path.home())
            data["home_disk_free_gb"] = round(usage.free / 1024**3, 1)
            data["home_disk_total_gb"] = round(usage.total / 1024**3, 1)
        except Exception:  # noqa: BLE001
            pass

        try:
            data["cpu_count"] = os.cpu_count()
            data["load_average"] = [round(v, 2) for v in os.getloadavg()]
        except Exception:  # noqa: BLE001
            pass

        for battery in sorted(Path("/sys/class/power_supply").glob("BAT*")):
            try:
                data["battery_percent"] = int(
                    (battery / "capacity").read_text().strip()
                )
                data["battery_status"] = (battery / "status").read_text().strip()
            except Exception:  # noqa: BLE001
                pass
            break

        lines = [f"{key.replace('_', ' ')}: {value}" for key, value in data.items()]
        return ToolResult.success("\n".join(lines), data=data)


def desktop_tools() -> list[BaseTool]:
    return [
        ListApplicationsTool(),
        OpenApplicationTool(),
        OpenUrlTool(),
        NotifyTool(),
        ClipboardReadTool(),
        ClipboardWriteTool(),
        MediaControlTool(),
        VolumeTool(),
        SystemInfoTool(),
    ]
