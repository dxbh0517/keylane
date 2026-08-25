"""The working orb — a corner overlay that spins, then opens into an answer.

Submitting from the popup should not hold you hostage while a worker runs. The
popup closes, a small orb appears in a screen corner and spins, and when the
answer lands the orb grows into a squircle showing the canvas.

Motion follows Apple's fluid-interface rules:

* the panel **expands from the orb**, not from the screen centre, so the
  spatial relationship is obvious (enter and exit share one path);
* the growth is a critically damped spring — ``response`` 0.4, no overshoot,
  because nothing here carried momentum;
* it is **interruptible** — clicking or pressing Escape mid-expansion collapses
  from wherever the animation currently is, never after finishing first.

Placement is the same problem the popup has: GNOME will not let a client
position a window. So the overlay is a screen-sized transparent window whose
*input region* is clipped to the visible orb, letting every click outside it
reach whatever is underneath.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
import threading
from typing import Any, Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

from gi.repository import Gdk, GLib, Gtk  # noqa: E402

from launcher.canvas_view import build_canvas  # noqa: E402
from launcher.loader import OrbitLoader  # noqa: E402

logger = logging.getLogger(__name__)

try:
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk4LayerShell as LayerShell  # type: ignore

    HAVE_LAYER_SHELL = True
except (ValueError, ImportError):  # pragma: no cover - environment dependent
    LayerShell = None  # type: ignore
    HAVE_LAYER_SHELL = False


ORB_SIZE = 56
# A one-line answer in a 440px panel is mostly empty panel. Width is chosen
# from how much there is to show, so the squircle fits its content.
PANEL_WIDTH = 440
PANEL_WIDTH_COMPACT = 300
COMPACT_CHARS = 90
PANEL_MAX_HEIGHT = 560
EDGE_MARGIN = 24

# The panel closes itself, because an answer you have read should not need
# dismissing. Hovering pauses the countdown — reading is a reason to stay.
DISMISS_AFTER = 9.0
DISMISS_AFTER_LONG = 20.0        # more to read, more time to read it
LONG_ANSWER_CHARS = 220
DISMISS_TICK_MS = 100

# Apple's move/reposition spring: critically damped, response 0.4s.
SPRING_RESPONSE = 0.36
SPRING_FPS = 60

CORNERS = {
    "top-right": (Gtk.Align.END, Gtk.Align.START),
    "top-left": (Gtk.Align.START, Gtk.Align.START),
    "bottom-right": (Gtk.Align.END, Gtk.Align.END),
    "bottom-left": (Gtk.Align.START, Gtk.Align.END),
    "center": (Gtk.Align.CENTER, Gtk.Align.CENTER),
}


def _canvas_text(canvas: dict[str, Any] | None) -> str:
    """Flatten a canvas into something worth reading aloud.

    Tables and code are described rather than spelled out — reading a df
    listing cell by cell is worse than useless.
    """
    if not canvas:
        return ""
    parts: list[str] = []
    for key in ("title", "summary"):
        value = str(canvas.get(key) or "").strip()
        if value:
            parts.append(value)
    for block in canvas.get("blocks") or []:
        kind = str(block.get("type") or "text")
        if kind in {"text", "heading", "note"}:
            parts.append(str(block.get("text") or ""))
        elif kind == "stats":
            parts += [
                f"{item.get('label')}: {item.get('value')}"
                for item in block.get("items") or []
            ]
        elif kind == "list":
            parts += [str(entry) for entry in block.get("entries") or []]
        elif kind == "table":
            rows = block.get("rows") or []
            parts.append(f"A table of {len(rows)} row{'s' if len(rows) != 1 else ''}.")
        elif kind == "code":
            parts.append("Command output is shown on screen.")
    cleaned: list[str] = []
    for part in parts:
        value = part.strip()
        if value:
            # Joining with ". " after a sentence that already ends in one
            # produces "..", which every synthesiser reads as a pause and a
            # stumble.
            cleaned.append(value.rstrip(".").strip() if value.endswith(".") else value)
    return ". ".join(cleaned)


def _panel_width_for(canvas: dict[str, Any] | None, text: str) -> int:
    """Wide enough for the content, narrow enough not to look empty."""
    blocks = (canvas or {}).get("blocks") or []
    kinds = {str(b.get("type") or "") for b in blocks}
    # Tables, stat rows and code all need horizontal room; prose does not.
    if kinds & {"table", "stats", "code", "links"}:
        return PANEL_WIDTH
    if len(blocks) > 2 or len(text) > COMPACT_CHARS:
        return PANEL_WIDTH
    return PANEL_WIDTH_COMPACT


def _critically_damped(t: float) -> float:
    """Unit step response of a critically damped spring, 0 → 1.

    ``1 - (1 + wt)e^(-wt)`` — settles without overshoot, which is what Apple
    prescribes for motion the user did not throw.
    """
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    w = 6.0  # reaches ~99% at t = 1
    x = w * t
    return 1.0 - (1.0 + x) * math.exp(-x)


class ResultOrb(Gtk.ApplicationWindow):
    """A corner overlay: spinner while working, canvas when done."""

    def __init__(
        self,
        app: Gtk.Application,
        *,
        client: Any = None,
        corner: str = "top-right",
        on_open_link: Callable[[str], None] | None = None,
        on_reopen: Callable[[], None] | None = None,
        on_followup: Callable[[str, str | None], None] | None = None,
    ) -> None:
        super().__init__(application=app, title="Keylane result")
        self.client = client
        self.corner = corner if corner in CORNERS else "top-right"
        self._speech_available = bool(client and client.speech_available())
        self._on_open_link = on_open_link
        self._on_reopen = on_reopen
        self._on_followup = on_followup

        self._expanded = False
        self._progress = 0.0        # 0 = orb, 1 = full panel
        self._target = 0.0
        self._tick: int | None = None
        self._canvas: dict[str, Any] | None = None
        self._actions: Gtk.Widget | None = None
        self._status = "Working…"
        self._panel_width = PANEL_WIDTH
        self._hovered = False
        self._dismiss_tick: int | None = None
        self._dismiss_left = 0.0
        self._speaking = False
        self._answer_text = ""
        self._session_id: str | None = None
        self._reply_row: Gtk.Widget | None = None

        self.set_decorated(False)
        self.set_resizable(False)
        self.set_hide_on_close(True)
        self.add_css_class("keylane-popup")
        self.add_css_class("keylane-orb-window")

        self._setup_layer_shell()
        self._build()

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self.add_controller(key)

        hover = Gtk.EventControllerMotion()
        hover.connect("enter", self._on_hover_enter)
        hover.connect("leave", self._on_hover_leave)
        self.add_controller(hover)
        self.connect("map", self._on_mapped)

    def _on_mapped(self, *_args) -> None:
        GLib.idle_add(self._update_input_region)
        # The X11 hint only applies once the surface exists.
        GLib.idle_add(self._keep_above)

    # ------------------------------------------------------------- placement

    def _setup_layer_shell(self) -> None:
        # Set first: every later branch may return early, and _build() reads it.
        self._layer_shell_active = False
        if not HAVE_LAYER_SHELL:
            return
        # Calling into layer-shell on an X11 display trips a GTK assertion.
        display = Gdk.Display.get_default()
        if display is None or "Wayland" not in type(display).__name__:
            return
        try:
            if not LayerShell.is_supported():  # type: ignore[union-attr]
                return
            LayerShell.init_for_window(self)  # type: ignore[union-attr]
            LayerShell.set_layer(self, LayerShell.Layer.OVERLAY)  # type: ignore[union-attr]
            LayerShell.set_namespace(self, "keylane-result")  # type: ignore[union-attr]
            LayerShell.set_keyboard_mode(  # type: ignore[union-attr]
                self, LayerShell.KeyboardMode.ON_DEMAND
            )
            top = self.corner.startswith("top")
            right = self.corner.endswith("right")
            LayerShell.set_anchor(self, LayerShell.Edge.TOP, top)  # type: ignore[union-attr]
            LayerShell.set_anchor(self, LayerShell.Edge.BOTTOM, not top)  # type: ignore[union-attr]
            LayerShell.set_anchor(self, LayerShell.Edge.RIGHT, right)  # type: ignore[union-attr]
            LayerShell.set_anchor(self, LayerShell.Edge.LEFT, not right)  # type: ignore[union-attr]
            for edge in (
                LayerShell.Edge.TOP,
                LayerShell.Edge.BOTTOM,
                LayerShell.Edge.LEFT,
                LayerShell.Edge.RIGHT,
            ):
                LayerShell.set_margin(self, edge, EDGE_MARGIN)  # type: ignore[union-attr]
            self._layer_shell_active = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("layer-shell unavailable for the orb: %s", exc)
            self._layer_shell_active = False

    def _keep_above(self) -> None:
        """Ask the window manager to stack this above everything else.

        Wayland gives clients no say in stacking, and Mutter does not
        implement layer-shell, so on GNOME the only working route is the X11
        one: run the surface through XWayland and set the standard
        ``_NET_WM_STATE_ABOVE`` hint, which Mutter does honour. Compositors
        with layer-shell never reach here — they got an overlay layer already.
        """
        if self._layer_shell_active:
            return
        surface = self.get_surface()
        if surface is None or not hasattr(surface, "get_xid"):
            return  # a Wayland toplevel: nothing more we can do
        try:
            xid = surface.get_xid()
        except Exception:  # noqa: BLE001
            return
        if not xid:
            return

        wmctrl = shutil.which("wmctrl")
        if wmctrl:
            subprocess.Popen(
                [wmctrl, "-i", "-r", str(xid), "-b", "add,above,sticky,skip_taskbar"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        xprop = shutil.which("xprop")
        if xprop:
            subprocess.Popen(
                [xprop, "-id", str(xid), "-f", "_NET_WM_STATE", "32a",
                 "-set", "_NET_WM_STATE", "_NET_WM_STATE_ABOVE"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            logger.info(
                "Cannot pin the result panel above other windows: install "
                "wmctrl, or run on a compositor with layer-shell."
            )

    def _screen_size(self) -> tuple[int, int]:
        display = Gdk.Display.get_default()
        if display is None:
            return 1920, 1080
        monitors = display.get_monitors()
        monitor = monitors.get_item(0) if monitors.get_n_items() else None
        if monitor is None:
            return 1920, 1080
        geometry = monitor.get_geometry()
        return geometry.width, geometry.height

    # ---------------------------------------------------------------- build

    def _build(self) -> None:
        self._shell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self._shell.add_css_class("keylane-shell")
        self._shell.add_css_class("keylane-result-shell")

        # Header: spinner or title, always present so the two states share a row.
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        header.set_valign(Gtk.Align.CENTER)

        self._loader = OrbitLoader(26)
        header.append(self._loader)

        self._status_label = Gtk.Label(label=self._status)
        self._status_label.set_xalign(0.0)
        self._status_label.set_hexpand(True)
        self._status_label.add_css_class("keylane-progress")
        self._status_label.set_ellipsize(3)  # END
        header.append(self._status_label)

        # Read aloud sits beside close, hidden until there is an answer and
        # speech is actually available.
        self._speak_btn = Gtk.Button()
        self._speak_btn.set_icon_name("audio-speakers-symbolic")
        self._speak_btn.set_tooltip_text("Read aloud")
        self._speak_btn.add_css_class("keylane-icon-btn")
        self._speak_btn.set_valign(Gtk.Align.CENTER)
        self._speak_btn.set_visible(False)
        self._speak_btn.connect("clicked", self._on_speak)
        header.append(self._speak_btn)

        self._close_btn = Gtk.Button()
        self._close_btn.set_icon_name("window-close-symbolic")
        self._close_btn.add_css_class("keylane-icon-btn")
        self._close_btn.set_valign(Gtk.Align.CENTER)
        self._close_btn.set_visible(False)
        self._close_btn.connect("clicked", lambda *_: self.dismiss())
        header.append(self._close_btn)

        self._header = header
        self._shell.append(header)

        self._body = Gtk.ScrolledWindow()
        self._body.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._body.set_propagate_natural_height(True)
        self._body.set_visible(False)
        self._body.add_css_class("keylane-result-view")
        self._shell.append(self._body)

        # Follow-up reply: stays with the answer so the conversation continues
        # without reopening the Spotlight bar.
        reply_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        reply_row.add_css_class("keylane-followup")
        reply_row.set_visible(False)
        self._reply_entry = Gtk.Entry()
        self._reply_entry.set_placeholder_text("Ask a follow-up…")
        self._reply_entry.set_hexpand(True)
        self._reply_entry.connect("activate", self._on_followup_submit)
        reply_row.append(self._reply_entry)
        send = Gtk.Button(label="Send")
        send.add_css_class("suggested-action")
        send.connect("clicked", self._on_followup_submit)
        reply_row.append(send)
        clear = Gtk.Button(label="Clear")
        clear.set_tooltip_text("Start a new conversation")
        clear.connect("clicked", self._on_clear_session)
        reply_row.append(clear)
        self._reply_row = reply_row
        self._shell.append(reply_row)

        halign, valign = CORNERS[self.corner]
        self._shell.set_halign(halign)
        self._shell.set_valign(valign)

        if self._layer_shell_active:
            self.set_child(self._shell)
        else:
            # No layer shell: cover the screen with a transparent window and
            # anchor the shell to the requested corner. The input region below
            # keeps the rest of the surface click-through.
            width, height = self._screen_size()
            self.set_default_size(width, height)
            # Explicitly not maximised: the window is screen-sized only so the
            # shell can sit in a corner, and a maximised state makes the WM
            # treat it as a real window.
            self.unmaximize()
            pad = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            pad.set_margin_top(EDGE_MARGIN)
            pad.set_margin_bottom(EDGE_MARGIN)
            pad.set_margin_start(EDGE_MARGIN)
            pad.set_margin_end(EDGE_MARGIN)
            pad.append(self._shell)
            self._shell.set_vexpand(False)
            pad.set_halign(Gtk.Align.FILL)
            pad.set_valign(Gtk.Align.FILL)
            self.set_child(pad)

        self._apply_geometry()

    # ------------------------------------------------------------- geometry

    def _apply_geometry(self) -> None:
        """Size the shell for the current point in the orb -> panel animation.

        Width interpolates; height is left to the content. Forcing a fixed
        panel height is what left a one-line answer floating in 40px of empty
        space — a squircle should be as tall as what it holds.
        """
        t = self._progress
        collapsed = t < 0.02

        if collapsed:
            self._shell.set_size_request(ORB_SIZE, ORB_SIZE)
            self._shell.add_css_class("is-orb")
        else:
            width = int(ORB_SIZE + (self._panel_width - ORB_SIZE) * t)
            self._shell.set_size_request(width, -1)
            self._shell.remove_css_class("is-orb")

        self._shell.set_hexpand(False)
        has_answer = self._canvas is not None

        # In the collapsed orb the loader is the only thing there, centred.
        self._loader.set_visible(collapsed or not has_answer)
        self._loader.set_size(26 if not collapsed else 30)
        # Collapsed, the header holds only the loader and must fill the orb so
        # CENTER alignment has the whole circle to centre within. Expanded, it
        # is a normal left-aligned row again.
        self._header.set_hexpand(collapsed)
        self._header.set_vexpand(collapsed)
        self._loader.set_hexpand(collapsed)
        self._loader.set_vexpand(collapsed)
        self._status_label.set_visible(not collapsed)
        self._close_btn.set_visible(t > 0.9 and has_answer)
        self._speak_btn.set_visible(t > 0.9 and has_answer and self._speech_available)

        self._body.set_visible(t > 0.55 and has_answer)
        self._body.set_opacity(max(0.0, min((t - 0.55) / 0.45, 1.0)))
        self._shell.set_opacity(0.35 + 0.65 * min(t * 4, 1.0) if t < 0.25 else 1.0)
        GLib.idle_add(self._update_input_region)

    def _animate_to(self, target: float) -> None:
        """Spring the shell towards ``target``, interruptibly.

        Starts from the *current* value, so grabbing a half-open panel and
        closing it never jumps.
        """
        self._target = target
        if self._tick is not None:
            return  # a frame loop is already running; it will chase the new target

        start = self._progress
        elapsed = 0.0
        step = 1.0 / SPRING_FPS

        def frame() -> bool:
            nonlocal start, elapsed
            elapsed += step
            fraction = _critically_damped(elapsed / SPRING_RESPONSE)
            self._progress = start + (self._target - start) * fraction
            self._apply_geometry()
            if fraction >= 0.999:
                self._progress = self._target
                self._apply_geometry()
                self._tick = None
                return False
            return True

        self._tick = GLib.timeout_add(int(1000 / SPRING_FPS), frame)

    # ------------------------------------------------------- auto-dismiss

    def _on_hover_enter(self, *_args) -> None:
        self._hovered = True
        self._shell.add_css_class("hovered")

    def _on_hover_leave(self, *_args) -> None:
        self._panel_width = PANEL_WIDTH
        self._hovered = False
        self._shell.remove_css_class("hovered")

    def _start_countdown(self, seconds: float) -> None:
        """Close after ``seconds``, pausing while the pointer is over it."""
        self._cancel_countdown()
        self._dismiss_left = seconds

        def tick() -> bool:
            # Hovering, speaking, typing a follow-up, or an unanswered approval
            # all mean the user is still using this — hold.
            focused = False
            try:
                focused = bool(
                    self._reply_entry is not None and self._reply_entry.has_focus()
                )
            except Exception:  # noqa: BLE001
                focused = False
            if self._hovered or self._speaking or self._actions is not None or focused:
                return True
            self._dismiss_left -= DISMISS_TICK_MS / 1000.0
            if self._dismiss_left <= 0:
                self._dismiss_tick = None
                self.dismiss()
                return False
            return True

        self._dismiss_tick = GLib.timeout_add(DISMISS_TICK_MS, tick)

    def _cancel_countdown(self) -> None:
        if self._dismiss_tick is not None:
            GLib.source_remove(self._dismiss_tick)
            self._dismiss_tick = None

    # -------------------------------------------------------- read aloud

    def _on_speak(self, *_args) -> None:
        if self._speaking:
            self._speak_btn.set_sensitive(False)
            threading.Thread(target=self._stop_speech, daemon=True).start()
            return
        text = self._answer_text.strip()
        if not text:
            return
        self._speaking = True
        self._loader.set_state("speaking")
        self._speak_btn.add_css_class("speaking")
        self._speak_btn.set_tooltip_text("Stop reading")

        def work() -> None:
            ok, detail = self.client.speak(text)

            def done() -> bool:
                self._speaking = False
                self._speak_btn.remove_css_class("speaking")
                self._speak_btn.set_tooltip_text("Read aloud")
                self._speak_btn.set_sensitive(True)
                if not ok and detail:
                    self._status_label.set_text(detail[:110])
                # Reading finished, so the countdown may resume.
                if self._canvas is not None and self._dismiss_tick is None:
                    self._start_countdown(DISMISS_AFTER)
                return False

            GLib.idle_add(done)

        threading.Thread(target=work, daemon=True).start()

    def _stop_speech(self) -> None:
        self.client.stop_speech()

    # ----------------------------------------------------------------- API

    def start(self, message: str) -> None:
        """Show the orb, spinning, for a request that has just been sent."""
        self._canvas = None
        self._answer_text = ""
        self._clear_actions()
        self._cancel_countdown()
        self._expanded = False
        self._progress = 0.0
        self._status = message[:80] or "Working…"
        self._status_label.set_text(self._status)
        self._set_body(None)
        if self._reply_row is not None:
            self._reply_row.set_visible(False)
            self._reply_entry.set_text("")
        self._apply_geometry()
        self._loader.set_state("thinking")
        self._loader.start()
        self.present()
        GLib.idle_add(self._keep_above)

    def set_status(self, text: str, state: str = "") -> None:
        """Update the caption and, with it, the colour of the loader."""
        self._status = text[:120]
        self._status_label.set_text(self._status)
        if state:
            self._loader.set_state(state)

    def set_state(self, state: str) -> None:
        self._loader.set_state(state)

    def set_session_id(self, session_id: str | None) -> None:
        self._session_id = session_id

    def show_result(
        self, canvas: dict[str, Any] | None, *, title: str = "", failed: bool = False
    ) -> None:
        """Expand the orb into the answer."""
        self._loader.stop()
        self._clear_actions()
        self._canvas = canvas
        self._answer_text = _canvas_text(canvas)
        self._panel_width = _panel_width_for(canvas, self._answer_text)
        self._status_label.set_text(
            title or (canvas or {}).get("title") or ("Failed" if failed else "Done")
        )
        self._loader.set_state("failed" if failed else "done")
        if failed:
            self._shell.add_css_class("failed")
        else:
            self._shell.remove_css_class("failed")
        self._set_body(canvas)
        if self._reply_row is not None and self._on_followup is not None and not failed:
            self._reply_row.set_visible(True)
        self._expanded = True
        self._animate_to(1.0)
        self.present()
        GLib.idle_add(self._keep_above)

        # A longer answer earns a longer read before it closes itself.
        # Follow-up entry pauses auto-dismiss while focused.
        self._start_countdown(
            DISMISS_AFTER_LONG
            if len(self._answer_text) > LONG_ANSWER_CHARS
            else DISMISS_AFTER
        )

    def _on_followup_submit(self, *_args) -> None:
        text = (self._reply_entry.get_text() or "").strip()
        if not text or self._on_followup is None:
            return
        self._cancel_countdown()
        self._reply_entry.set_text("")
        self._on_followup(text, self._session_id)

    def _on_clear_session(self, *_args) -> None:
        session_id = self._session_id
        self._session_id = None
        if session_id and self.client is not None:
            def work() -> None:
                try:
                    self.client.clear_session(session_id)
                except Exception:  # noqa: BLE001
                    logger.debug("clear_session failed", exc_info=True)

            threading.Thread(target=work, daemon=True).start()
        self._status_label.set_text("New conversation")
        if self._reply_entry is not None:
            self._reply_entry.set_placeholder_text("Ask something new…")

    def show_confirmation(
        self,
        data: dict[str, Any],
        *,
        on_allow: Callable[[], None],
        on_cancel: Callable[[], None],
    ) -> None:
        """Expand with an Allow / Cancel choice instead of an answer.

        A gated tool has to be approvable without reopening the bar, or the
        hand-off to the orb would just lose the task.
        """
        self._loader.stop()
        self._loader.set_state("waiting")
        self._cancel_countdown()
        tool = data.get("pending_tool") or "this action"
        arguments = data.get("pending_arguments") or {}
        self._status_label.set_text("Approval needed")

        canvas: dict[str, Any] = {
            "blocks": [
                {"type": "text", "text": f"Keylane wants to run {tool}."},
            ]
        }
        if arguments:
            canvas["blocks"].append(
                {
                    "type": "table",
                    "columns": ["Argument", "Value"],
                    "rows": [[str(k), str(v)] for k, v in arguments.items()],
                }
            )
        self._canvas = canvas
        self._panel_width = PANEL_WIDTH
        self._set_body(canvas)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.set_halign(Gtk.Align.END)
        cancel = Gtk.Button(label="Cancel")
        cancel.connect("clicked", lambda *_: (on_cancel(), self._clear_actions()))
        row.append(cancel)
        allow = Gtk.Button(label="Allow")
        allow.add_css_class("suggested-action")
        allow.connect("clicked", lambda *_: (self._clear_actions(), on_allow()))
        row.append(allow)

        self._clear_actions()
        self._actions = row
        self._shell.append(row)

        self._expanded = True
        self._animate_to(1.0)
        self.present()

    def _clear_actions(self) -> None:
        actions = getattr(self, "_actions", None)
        if actions is not None:
            self._shell.remove(actions)
            self._actions = None

    def _set_body(self, canvas: dict[str, Any] | None) -> None:
        widget = build_canvas(canvas, on_open=self._on_open_link)
        self._body.set_child(widget)
        self._body.set_max_content_height(PANEL_MAX_HEIGHT)

    def dismiss(self) -> None:
        """Collapse and hide. Interruptible at any point in the animation."""
        self._loader.stop()
        self._cancel_countdown()
        self._expanded = False
        self._animate_to(0.0)

        def hide() -> bool:
            if not self._expanded:
                self.set_visible(False)
                self._progress = 0.0
                self._apply_geometry()
            return False

        GLib.timeout_add(int(SPRING_RESPONSE * 1000) + 40, hide)

    # -------------------------------------------------------------- events

    def _on_key(self, _controller, keyval, _keycode, _state) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.dismiss()
            return True
        return False

    def _update_input_region(self) -> None:
        """Only the visible shell should catch clicks."""
        surface = self.get_surface()
        if surface is None:
            return
        try:
            import cairo

            ok, rect = self._shell.compute_bounds(self)
            if not ok:
                return
            region = cairo.Region(
                cairo.RectangleInt(
                    int(rect.origin.x),
                    int(rect.origin.y),
                    max(int(rect.size.width), 1),
                    max(int(rect.size.height), 1),
                )
            )
            surface.set_input_region(region)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not clip the orb input region: %s", exc)
