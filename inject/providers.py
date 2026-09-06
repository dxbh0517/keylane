"""The four ways to put text in someone else's window.

Which one runs is a property of the session, not a preference — the same shape
as ``runtimes/`` picking a backend from what an export on disk actually is:

* **wtype** speaks ``zwp_virtual_keyboard_v1`` and is the right answer on
  wlroots compositors (Sway, Hyprland, river).
* **ydotool** goes underneath the display server through ``uinput``. It is the
  *only* route on GNOME, because Mutter has never implemented the
  virtual-keyboard protocol — the same gap that forces Keylane onto XWayland
  for placement. It needs ``ydotoold`` running and the user in a group that
  can open ``/dev/uinput``.
* **xdotool** covers a real X11 session.
* **clipboard** is the universal fallback: put the text on the clipboard and
  press paste. It needs one of the three above to press the keys, so it is a
  fallback for *layout* trouble rather than for having no injector at all.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

from inject.base import (
    InjectionError,
    InjectionTarget,
    run_with_text,
)

logger = logging.getLogger(__name__)

# Linux input event codes, used by ydotool. Spelled out because ydotool's
# friendlier `ctrl+v` spelling only exists in newer builds.
_KEY_LEFTCTRL = 29
_KEY_LEFTSHIFT = 42
_KEY_V = 47

# How long to leave the text on the clipboard before putting back whatever the
# user had. Long enough for the target to service the paste, short enough that
# a Ctrl+V of their own a moment later still gets their content.
CLIPBOARD_RESTORE_DELAY = 1.0


class WtypeProvider:
    name = "wtype"
    layout_safe = False

    def available(self) -> bool:
        return shutil.which("wtype") is not None

    def type_text(self, text: str) -> None:
        run_with_text(["wtype", "-"], text, name="wtype")

    def paste(self, *, shifted: bool) -> None:
        mods = ["-M", "ctrl"] + (["-M", "shift"] if shifted else [])
        release = ["-m", "ctrl"] + (["-m", "shift"] if shifted else [])
        run_with_text(["wtype", *mods, "v", *release], "", name="wtype")


class YdotoolProvider:
    name = "ydotool"
    layout_safe = False

    def available(self) -> bool:
        if shutil.which("ydotool") is None:
            return False
        # ydotool without its daemon fails at the point of use with a message
        # about the socket, which is a confusing way to learn it is not set up.
        socket_path = os.environ.get(
            "YDOTOOL_SOCKET", f"/run/user/{os.getuid()}/.ydotool_socket"
        )
        return os.path.exists(socket_path) or os.path.exists("/tmp/.ydotool_socket")

    def type_text(self, text: str) -> None:
        run_with_text(["ydotool", "type", "--file", "-"], text, name="ydotool")

    def paste(self, *, shifted: bool) -> None:
        held = [_KEY_LEFTCTRL] + ([_KEY_LEFTSHIFT] if shifted else [])
        sequence = (
            [f"{code}:1" for code in held]
            + [f"{_KEY_V}:1", f"{_KEY_V}:0"]
            + [f"{code}:0" for code in reversed(held)]
        )
        run_with_text(["ydotool", "key", *sequence], "", name="ydotool")


class XdotoolProvider:
    name = "xdotool"
    layout_safe = False

    def available(self) -> bool:
        return shutil.which("xdotool") is not None and bool(os.environ.get("DISPLAY"))

    def type_text(self, text: str) -> None:
        run_with_text(
            ["xdotool", "type", "--clearmodifiers", "--file", "-"], text, name="xdotool"
        )

    def paste(self, *, shifted: bool) -> None:
        combo = "ctrl+shift+v" if shifted else "ctrl+v"
        run_with_text(["xdotool", "key", "--clearmodifiers", combo], "", name="xdotool")


# ── the clipboard route ──────────────────────────────────────────────────


def _clipboard_tools() -> tuple[list[str], list[str]] | None:
    """``(copy_argv, paste_argv)`` for this session, or None if neither exists."""
    if shutil.which("wl-copy") and shutil.which("wl-paste"):
        return ["wl-copy"], ["wl-paste", "--no-newline"]
    if shutil.which("xclip"):
        return (
            ["xclip", "-selection", "clipboard", "-in"],
            ["xclip", "-selection", "clipboard", "-out"],
        )
    return None


def read_clipboard() -> str:
    tools = _clipboard_tools()
    if tools is None:
        return ""
    try:
        done = subprocess.run(tools[1], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout if done.returncode == 0 else ""


def write_clipboard(text: str) -> None:
    tools = _clipboard_tools()
    if tools is None:
        raise InjectionError("no clipboard tool found (install wl-clipboard or xclip)")
    run_with_text(tools[0], text, name=tools[0][0])


class ClipboardProvider:
    """Copy, press paste, then put the user's clipboard back.

    Restoring matters more than it looks: dictating three sentences would
    otherwise silently destroy whatever the user had copied, three times.
    """

    name = "clipboard"
    layout_safe = True

    def __init__(self, presser: object | None = None) -> None:
        self._presser = presser

    def _key_provider(self):
        if self._presser is not None:
            return self._presser
        for candidate in (WtypeProvider(), YdotoolProvider(), XdotoolProvider()):
            if candidate.available():
                return candidate
        return None

    def available(self) -> bool:
        return _clipboard_tools() is not None and self._key_provider() is not None

    def type_text(self, text: str, target: InjectionTarget | None = None) -> None:
        presser = self._key_provider()
        if presser is None:
            raise InjectionError("nothing available to press paste with")
        previous = read_clipboard()
        write_clipboard(text)
        try:
            # A terminal takes Ctrl+Shift+V; Ctrl+V there is often a no-op or,
            # worse, a literal control character.
            presser.paste(shifted=bool(target and target.is_terminal))
        finally:
            if previous:
                time.sleep(CLIPBOARD_RESTORE_DELAY)
                try:
                    write_clipboard(previous)
                except InjectionError:
                    logger.warning("could not restore the previous clipboard contents")
