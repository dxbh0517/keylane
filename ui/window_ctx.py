"""Which window has the keyboard, and where it is on screen.

Four routes, and the fourth is an admission rather than a fallback:

* **Sway / river** answer ``swaymsg -t get_tree`` with the whole tree; the
  focused node is the one with ``focused: true``.
* **Hyprland** answers ``hyprctl -j activewindow`` directly.
* **X11 and XWayland** keep ``_NET_ACTIVE_WINDOW`` on the root window, which
  is also how Keylane already reaches ``wmctrl`` for placement.
* **GNOME on Wayland** exposes no supported way for an ordinary client to learn
  the focused native-Wayland client's identity. Mutter deliberately does not
  ship one, and the Shell-eval route is closed on any current version. There
  the answer is :data:`UNKNOWN`, and callers must degrade rather than guess —
  an app name invented here would silently inject the wrong app skill.

The geometry is in *logical* pixels, the same units ``ui/placement.py`` works
in, so an annotation drawn from it lands where the window is.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_TIMEOUT = 3

# Keylane's own surfaces. Dictation triggered from a hotkey should target the
# app the user was in, and on a compositor that focuses the answer HUD that is
# not the same thing as "whatever is focused right now".
_OWN_WINDOW_HINTS = ("keylane", "spotlight")


@dataclass(frozen=True)
class FocusedWindow:
    """The focused window, as far as this display server will say."""

    app_id: str = ""
    title: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    source: str = "unknown"

    @property
    def known(self) -> bool:
        return bool(self.app_id or self.title)

    @property
    def is_keylane(self) -> bool:
        haystack = f"{self.app_id} {self.title}".lower()
        return any(hint in haystack for hint in _OWN_WINDOW_HINTS)

    def label(self) -> str:
        """A short human name for the HUD — "Firefox", not a window class."""
        if self.app_id:
            leaf = self.app_id.rsplit(".", 1)[-1]
            return leaf.replace("-", " ").strip() or self.app_id
        if self.title:
            return self.title[:40]
        return "unknown app"


UNKNOWN = FocusedWindow()


def _run(argv: list[str]) -> str:
    try:
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=_TIMEOUT, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout if done.returncode == 0 else ""


# ── sway / river ─────────────────────────────────────────────────────────


def _focused_node(node: dict) -> dict | None:
    """Depth-first search for the focused leaf of a sway tree."""
    if node.get("focused") and node.get("type") in {"con", "floating_con"}:
        return node
    for child in list(node.get("nodes", [])) + list(node.get("floating_nodes", [])):
        found = _focused_node(child)
        if found is not None:
            return found
    return None


def _from_sway() -> FocusedWindow | None:
    if not shutil.which("swaymsg") or not os.environ.get("SWAYSOCK"):
        return None
    raw = _run(["swaymsg", "-t", "get_tree", "-r"])
    if not raw:
        return None
    try:
        tree = json.loads(raw)
    except json.JSONDecodeError:
        return None
    node = _focused_node(tree)
    if node is None:
        return None
    rect = node.get("rect") or {}
    # A native Wayland client has app_id; an XWayland one has window_properties.
    app_id = node.get("app_id") or (node.get("window_properties") or {}).get("class") or ""
    return FocusedWindow(
        app_id=str(app_id),
        title=str(node.get("name") or ""),
        x=int(rect.get("x", 0)),
        y=int(rect.get("y", 0)),
        width=int(rect.get("width", 0)),
        height=int(rect.get("height", 0)),
        source="sway",
    )


# ── hyprland ─────────────────────────────────────────────────────────────


def _from_hyprland() -> FocusedWindow | None:
    if not shutil.which("hyprctl") or not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return None
    raw = _run(["hyprctl", "-j", "activewindow"])
    if not raw:
        return None
    try:
        window = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(window, dict) or not window:
        return None
    at = window.get("at") or [0, 0]
    size = window.get("size") or [0, 0]
    return FocusedWindow(
        app_id=str(window.get("class") or ""),
        title=str(window.get("title") or ""),
        x=int(at[0]) if len(at) > 0 else 0,
        y=int(at[1]) if len(at) > 1 else 0,
        width=int(size[0]) if len(size) > 0 else 0,
        height=int(size[1]) if len(size) > 1 else 0,
        source="hyprland",
    )


# ── X11 / XWayland ───────────────────────────────────────────────────────

_WINDOW_ID = re.compile(r"window id # (0x[0-9a-fA-F]+)")
_WM_CLASS = re.compile(r'WM_CLASS\(STRING\) = "([^"]*)"(?:, "([^"]*)")?')
_WM_NAME = re.compile(r"(?:_NET_WM_NAME|WM_NAME)\((?:UTF8_STRING|STRING)\) = \"(.*)\"")
_GEOM = re.compile(
    r"Absolute upper-left X:\s*(-?\d+).*?"
    r"Absolute upper-left Y:\s*(-?\d+).*?"
    r"Width:\s*(\d+).*?Height:\s*(\d+)",
    re.DOTALL,
)


def _from_x11() -> FocusedWindow | None:
    if not shutil.which("xprop") or not os.environ.get("DISPLAY"):
        return None
    root = _run(["xprop", "-root", "_NET_ACTIVE_WINDOW"])
    match = re.search(r"(0x[0-9a-fA-F]+)", root)
    if not match:
        return None
    win_id = match.group(1)
    if int(win_id, 16) == 0:
        return None

    props = _run(["xprop", "-id", win_id, "WM_CLASS", "_NET_WM_NAME", "WM_NAME"])
    class_match = _WM_CLASS.search(props)
    # WM_CLASS is (instance, class); the second is the one users recognise.
    app_id = ""
    if class_match:
        app_id = class_match.group(2) or class_match.group(1) or ""
    name_match = _WM_NAME.search(props)
    title = name_match.group(1) if name_match else ""

    x = y = width = height = 0
    if shutil.which("xwininfo"):
        geom = _GEOM.search(_run(["xwininfo", "-id", win_id]))
        if geom:
            x, y, width, height = (int(g) for g in geom.groups())

    return FocusedWindow(
        app_id=str(app_id),
        title=str(title),
        x=x,
        y=y,
        width=width,
        height=height,
        source="x11",
    )


# ── public API ───────────────────────────────────────────────────────────


def focused_window() -> FocusedWindow:
    """The focused window, or :data:`UNKNOWN` where the session will not say."""
    for probe in (_from_sway, _from_hyprland, _from_x11):
        try:
            found = probe()
        except Exception:  # noqa: BLE001
            logger.debug("focus probe %s failed", probe.__name__, exc_info=True)
            continue
        if found is not None and found.known:
            return found
    return UNKNOWN


def describe() -> dict[str, object]:
    """What Settings shows about this capability."""
    window = focused_window()
    return {
        "ok": window.known,
        "source": window.source,
        "app_id": window.app_id,
        "title": window.title,
        "reason": (
            ""
            if window.known
            else "This compositor does not tell ordinary clients which window has focus."
        ),
    }
