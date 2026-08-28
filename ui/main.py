#!/usr/bin/env python3
"""Keylane Spotlight — macOS-style floating command bar."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

_LAYER_SHELL_LIB = "/usr/lib64/libgtk4-layer-shell.so.0"
if (
    os.path.isfile(_LAYER_SHELL_LIB)
    and _LAYER_SHELL_LIB not in os.environ.get("LD_PRELOAD", "")
    and os.environ.get("KEYLANE_LAYER_SHELL_PRIMED") != "1"
):
    os.environ["LD_PRELOAD"] = _LAYER_SHELL_LIB + (
        f":{os.environ['LD_PRELOAD']}" if os.environ.get("LD_PRELOAD") else ""
    )
    os.environ["KEYLANE_LAYER_SHELL_PRIMED"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gio", "2.0")
try:
    gi.require_version("Gtk4LayerShell", "1.0")
    HAS_LAYER_SHELL = True
except ValueError:
    HAS_LAYER_SHELL = False

from gi.repository import Gdk, Gio, GLib, Gtk  # type: ignore[attr-defined]

if HAS_LAYER_SHELL:
    from gi.repository import Gtk4LayerShell as LayerShell  # type: ignore[attr-defined]

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.settings import SettingsWindow
from ui.theme import apply_scheme_classes, apply_spotlight_theme, watch_color_scheme
from ui.voice import record_and_transcribe

DAEMON = "http://127.0.0.1:9100"
PANEL_WIDTH = 680
CORNER_WIDTH = 380
CORNER_MARGIN = 20
logger = logging.getLogger(__name__)


def _layer_shell_ok() -> bool:
    if not HAS_LAYER_SHELL:
        return False
    try:
        return bool(LayerShell.is_supported())
    except Exception:  # noqa: BLE001
        return False


def _monitor_size() -> tuple[int, int]:
    display = Gdk.Display.get_default()
    if not display:
        return 1920, 1080
    monitors = display.get_monitors()
    monitor = monitors.get_item(0) if monitors.get_n_items() > 0 else None
    if monitor is None:
        return 1920, 1080
    geom = monitor.get_geometry()
    return geom.width, geom.height


def _configure_layer_shell_overlay(window: Gtk.ApplicationWindow) -> bool:
    if not _layer_shell_ok():
        return False
    try:
        LayerShell.init_for_window(window)
        if not LayerShell.is_layer_window(window):
            return False
        LayerShell.set_layer(window, LayerShell.Layer.OVERLAY)
        LayerShell.set_keyboard_mode(window, LayerShell.KeyboardMode.ON_DEMAND)
        for edge in (
            LayerShell.Edge.TOP,
            LayerShell.Edge.BOTTOM,
            LayerShell.Edge.LEFT,
            LayerShell.Edge.RIGHT,
        ):
            LayerShell.set_anchor(window, edge, True)
            LayerShell.set_margin(window, edge, 0)
        LayerShell.set_exclusive_zone(window, -1)
        return True
    except Exception:  # noqa: BLE001
        logger.warning("layer shell init failed", exc_info=True)
        return False


def _configure_layer_shell_corner(window: Gtk.ApplicationWindow) -> None:
    if not _layer_shell_ok() or not LayerShell.is_layer_window(window):
        return
    LayerShell.set_keyboard_mode(window, LayerShell.KeyboardMode.EXCLUSIVE)
    for edge in (LayerShell.Edge.BOTTOM, LayerShell.Edge.LEFT):
        LayerShell.set_anchor(window, edge, False)
        LayerShell.set_margin(window, edge, 0)
    for edge in (LayerShell.Edge.TOP, LayerShell.Edge.RIGHT):
        LayerShell.set_anchor(window, edge, True)
        LayerShell.set_margin(window, edge, CORNER_MARGIN)


def _configure_layer_shell_spotlight(window: Gtk.ApplicationWindow) -> None:
    if not _layer_shell_ok() or not LayerShell.is_layer_window(window):
        return
    LayerShell.set_keyboard_mode(window, LayerShell.KeyboardMode.ON_DEMAND)
    for edge in (
        LayerShell.Edge.TOP,
        LayerShell.Edge.BOTTOM,
        LayerShell.Edge.LEFT,
        LayerShell.Edge.RIGHT,
    ):
        LayerShell.set_anchor(window, edge, True)
        LayerShell.set_margin(window, edge, 0)


def _configure_fallback_window(window: Gtk.ApplicationWindow) -> None:
    width, height = _monitor_size()
    window.set_default_size(width, height)
    window.set_resizable(False)
    window.set_modal(True)


class SpotlightWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app)
        self.add_css_class("spotlight-window")
        self.set_decorated(False)
        self.set_deletable(False)
        self.session_id: str | None = None
        self._mode = "spotlight"
        self._busy = False
        self._streaming_answer = False
        self._sources: list[dict[str, str]] = []
        self._settings_win: SettingsWindow | None = None

        self._layered = _configure_layer_shell_overlay(self)
        if not self._layered:
            _configure_fallback_window(self)
        else:
            self.set_resizable(False)
            self.set_default_size(PANEL_WIDTH, 1)

        overlay = Gtk.Overlay()
        overlay.set_hexpand(True)
        overlay.set_vexpand(True)
        self.set_child(overlay)

        self._scrim = Gtk.Box()
        self._scrim.add_css_class("spotlight-scrim")
        self._scrim.set_hexpand(True)
        self._scrim.set_vexpand(True)
        overlay.set_child(self._scrim)

        click = Gtk.GestureClick.new()
        click.connect("released", self._on_scrim_click)
        self._scrim.add_controller(click)

        self._backdrop = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._backdrop.add_css_class("spotlight-backdrop")
        self._backdrop.set_halign(Gtk.Align.CENTER)
        self._backdrop.set_valign(Gtk.Align.CENTER)
        self._backdrop.set_margin_start(24)
        self._backdrop.set_margin_end(24)
        overlay.add_overlay(self._backdrop)

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._stack.set_transition_duration(200)
        self._stack.add_css_class("spotlight-stack")
        self._backdrop.append(self._stack)

        self._build_spotlight_page()
        self._build_corner_page()

        apply_scheme_classes(self)
        watch_color_scheme(lambda _dark: apply_scheme_classes(self))

        key = Gtk.EventControllerKey.new()
        key.connect("key-released", self._on_key)
        self.add_controller(key)

        self.set_visible(False)
        self._refresh_status()

    def _build_spotlight_page(self) -> None:
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        panel.add_css_class("spotlight-panel")
        panel.set_size_request(PANEL_WIDTH, -1)
        self._spotlight_panel = panel
        self._stack.add_named(panel, "spotlight")

        search_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        search_row.add_css_class("spotlight-search-row")
        panel.append(search_row)

        search_icon = Gtk.Image.new_from_icon_name("edit-find-symbolic")
        search_icon.set_pixel_size(20)
        search_icon.add_css_class("spotlight-search-icon")
        search_row.append(search_icon)

        self.entry = Gtk.Entry()
        self.entry.add_css_class("spotlight-search")
        self.entry.set_placeholder_text("Search or ask Keylane")
        self.entry.set_has_frame(False)
        self.entry.set_hexpand(True)
        self.entry.connect("activate", self._on_submit)
        search_row.append(self.entry)

        mic_btn = Gtk.Button()
        mic_btn.set_child(Gtk.Image.new_from_icon_name("audio-input-microphone-symbolic"))
        mic_btn.get_child().set_pixel_size(18)  # type: ignore[union-attr]
        mic_btn.set_tooltip_text("Voice input")
        mic_btn.add_css_class("spotlight-icon-btn")
        mic_btn.set_can_focus(False)
        mic_btn.connect("clicked", self._on_mic)
        search_row.append(mic_btn)

        hist_btn = Gtk.Button()
        hist_btn.set_child(Gtk.Image.new_from_icon_name("document-open-recent-symbolic"))
        hist_btn.get_child().set_pixel_size(18)  # type: ignore[union-attr]
        hist_btn.set_tooltip_text("History (Ctrl+H)")
        hist_btn.add_css_class("spotlight-icon-btn")
        hist_btn.set_can_focus(False)
        hist_btn.connect("clicked", lambda *_: self._toggle_history())
        search_row.append(hist_btn)

        self._history_revealer = Gtk.Revealer()
        self._history_revealer.set_reveal_child(False)
        panel.append(self._history_revealer)

        hist_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        hist_box.add_css_class("spotlight-history")
        self._history_revealer.set_child(hist_box)

        hist_scroll = Gtk.ScrolledWindow()
        hist_scroll.add_css_class("spotlight-history-scroll")
        hist_scroll.set_max_content_height(160)
        hist_box.append(hist_scroll)

        self._history_list = Gtk.ListBox()
        self._history_list.add_css_class("spotlight-history-list")
        self._history_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._history_list.connect("row-activated", self._on_history_pick)
        hist_scroll.set_child(self._history_list)

        footer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        footer.add_css_class("spotlight-footer")
        panel.append(footer)

        pill = Gtk.Label(label="Keylane")
        pill.add_css_class("spotlight-pill")
        footer.append(pill)

        self.status = Gtk.Label(label="")
        self.status.add_css_class("spotlight-status")
        self.status.set_hexpand(True)
        self.status.set_halign(Gtk.Align.END)
        footer.append(self.status)

        settings_btn = Gtk.Button()
        settings_btn.set_child(Gtk.Image.new_from_icon_name("preferences-system-symbolic"))
        settings_btn.get_child().set_pixel_size(16)  # type: ignore[union-attr]
        settings_btn.set_tooltip_text("Settings (Ctrl+,)")
        settings_btn.add_css_class("spotlight-footer-btn")
        settings_btn.set_can_focus(False)
        settings_btn.connect("clicked", lambda *_: self._open_settings())
        footer.append(settings_btn)

    def _build_corner_page(self) -> None:
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        panel.add_css_class("corner-panel")
        panel.set_size_request(CORNER_WIDTH, -1)
        self._corner_panel = panel
        self._stack.add_named(panel, "corner")

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.add_css_class("corner-header")
        panel.append(header)

        self._corner_title = Gtk.Label(label="Keylane")
        self._corner_title.add_css_class("corner-title")
        self._corner_title.set_halign(Gtk.Align.START)
        self._corner_title.set_hexpand(True)
        self._corner_title.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        self._corner_title.set_max_width_chars(42)
        header.append(self._corner_title)

        close_btn = Gtk.Button()
        close_btn.add_css_class("corner-close")
        close_btn.set_icon_name("window-close-symbolic")
        close_btn.set_tooltip_text("Dismiss")
        close_btn.connect("clicked", lambda *_: self._dismiss_corner())
        header.append(close_btn)

        progress_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        progress_row.add_css_class("corner-progress")
        panel.append(progress_row)

        self._spinner = Gtk.Spinner()
        self._spinner.add_css_class("corner-spinner")
        self._spinner.start()
        progress_row.append(self._spinner)

        self._corner_status = Gtk.Label(label="Thinking…")
        self._corner_status.add_css_class("corner-status")
        self._corner_status.set_halign(Gtk.Align.START)
        self._corner_status.set_hexpand(True)
        self._corner_status.set_wrap(True)
        progress_row.append(self._corner_status)

        self._answer_revealer = Gtk.Revealer()
        self._answer_revealer.set_reveal_child(False)
        self._answer_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._answer_revealer.set_transition_duration(220)
        panel.append(self._answer_revealer)

        answer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        answer_box.add_css_class("corner-answer-box")
        self._answer_revealer.set_child(answer_box)

        scroll = Gtk.ScrolledWindow()
        scroll.add_css_class("corner-scroll")
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_max_content_height(280)
        scroll.set_vexpand(False)
        answer_box.append(scroll)

        self._corner_answer = Gtk.TextView()
        self._corner_answer.add_css_class("corner-answer")
        self._corner_answer.set_editable(False)
        self._corner_answer.set_wrap_mode(Gtk.WrapMode.WORD)
        scroll.set_child(self._corner_answer)

        self._sources_revealer = Gtk.Revealer()
        self._sources_revealer.set_reveal_child(False)
        answer_box.append(self._sources_revealer)

        sources_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        sources_box.add_css_class("corner-sources")
        self._sources_revealer.set_child(sources_box)

        sources_box.append(Gtk.Label(label="Sources", xalign=0, css_classes=["corner-sources-title"]))
        self._sources_label = Gtk.Label(label="", xalign=0, wrap=True, css_classes=["corner-sources-list"])
        sources_box.append(self._sources_label)

    def _on_scrim_click(self, _gesture, _n_press: int, _x: float, _y: float) -> None:
        if self._mode == "spotlight" and not self._busy:
            self._hide_spotlight()

    def _on_key(self, _ctrl: Gtk.EventControllerKey, keyval: int, _keycode: int, _state: Gdk.ModifierType) -> bool:
        if keyval == Gdk.KEY_Escape:
            if self._mode == "corner":
                self._dismiss_corner()
            else:
                self._hide_spotlight()
            return True
        if keyval == Gdk.KEY_h and (_state & Gdk.ModifierType.CONTROL_MASK):
            self._toggle_history()
            return True
        if keyval in (Gdk.KEY_comma, Gdk.KEY_semicolon) and (_state & Gdk.ModifierType.CONTROL_MASK):
            self._open_settings()
            return True
        return False

    def _show_toast(self, message: str) -> None:
        self.status.set_text(message[:40])

    def _open_settings(self) -> None:
        try:
            if self._settings_win is None:
                self._settings_win = SettingsWindow(self, independent=self._layered)
                self._settings_win.set_toast_callback(self._show_toast)
                self._settings_win.set_scheme_callback(lambda: apply_scheme_classes(self))
            self._settings_win.load_settings()
            self._settings_win.present_centered()
        except Exception:  # noqa: BLE001
            logger.exception("failed to open settings")
            self._show_toast("Settings failed to open")

    def _toggle_history(self) -> None:
        reveal = not self._history_revealer.get_reveal_child()
        if reveal:
            self._load_history()
        self._history_revealer.set_reveal_child(reveal)

    def _load_history(self) -> None:
        child = self._history_list.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._history_list.remove(child)
            child = next_child
        try:
            rows = httpx.get(f"{DAEMON}/sessions", timeout=5).json().get("sessions", [])
        except Exception:  # noqa: BLE001
            return
        for row in rows:
            title = row.get("title") or "Chat"
            updated = (row.get("updated_at") or "")[:16]
            label = Gtk.Label(label=f"{title} · {updated}", xalign=0, ellipsize=3)
            label.add_css_class("spotlight-history-label")
            list_row = Gtk.ListBoxRow()
            list_row.add_css_class("spotlight-history-row")
            list_row.set_child(label)
            list_row.session_id = row["id"]  # type: ignore[attr-defined]
            self._history_list.append(list_row)

    def _on_history_pick(self, _list: Gtk.ListBox, row: Gtk.ListBoxRow) -> None:
        sid = getattr(row, "session_id", None)
        if sid:
            self.session_id = sid
            self._show_toast(f"Resumed session")
        self._history_revealer.set_reveal_child(False)

    def _on_mic(self, *_args) -> None:
        if self._busy:
            return
        self._show_toast("Listening…")

        def on_done(text: str) -> None:
            def _ui() -> bool:
                if text:
                    self.entry.set_text(text)
                    self._show_toast("Transcribed")
                else:
                    self._show_toast("No speech detected")
                return False

            GLib.idle_add(_ui)

        def on_error(msg: str) -> None:
            GLib.idle_add(self._show_toast, msg[:40])

        record_and_transcribe(on_done=on_done, on_error=on_error)

    def _show_permission_dialog(self, perm: dict) -> None:
        dialog = Gtk.MessageDialog(
            transient_for=self,
            modal=True,
            message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.NONE,
            text=f"Allow tool: {perm.get('tool', '?')}?",
            secondary_text=str(perm.get("arguments", {}))[:200],
        )
        allow = dialog.add_button("Allow", Gtk.ResponseType.ACCEPT)
        deny = dialog.add_button("Deny", Gtk.ResponseType.CANCEL)
        allow.grab_default()

        def _respond(_dlg, response: Gtk.ResponseType) -> None:
            approved = response == Gtk.ResponseType.ACCEPT
            try:
                httpx.post(
                    f"{DAEMON}/permissions/respond",
                    json={"id": perm.get("id"), "approved": approved},
                    timeout=5,
                )
            except Exception:  # noqa: BLE001
                pass
            dialog.destroy()

        dialog.connect("response", _respond)
        dialog.present()

    def _hide_spotlight(self) -> None:
        self._scrim.remove_css_class("visible")
        self.set_visible(False)

    def _dismiss_corner(self) -> None:
        self._busy = False
        self.remove_css_class("corner-mode")
        self._scrim.remove_css_class("visible")
        self._configure_spotlight_layout()
        self._stack.set_visible_child_name("spotlight")
        self._mode = "spotlight"
        self.entry.set_sensitive(True)
        self.set_visible(False)

    def _configure_spotlight_layout(self) -> None:
        self._backdrop.set_halign(Gtk.Align.CENTER)
        self._backdrop.set_valign(Gtk.Align.CENTER)
        self._backdrop.set_margin_top(0)
        self._backdrop.set_margin_bottom(0)
        self._backdrop.set_margin_start(24)
        self._backdrop.set_margin_end(24)
        if self._layered:
            _configure_layer_shell_spotlight(self)

    def _configure_corner_layout(self) -> None:
        self._backdrop.set_halign(Gtk.Align.END)
        self._backdrop.set_valign(Gtk.Align.START)
        self._backdrop.set_margin_top(CORNER_MARGIN)
        self._backdrop.set_margin_bottom(0)
        self._backdrop.set_margin_start(0)
        self._backdrop.set_margin_end(CORNER_MARGIN)
        if self._layered:
            _configure_layer_shell_corner(self)

    def _enter_corner_mode(self, query: str) -> None:
        self._mode = "corner"
        self._busy = True
        self.add_css_class("corner-mode")
        self._scrim.remove_css_class("visible")

        self._corner_title.set_text(query)
        self._corner_status.set_text("Thinking…")
        self._corner_answer.get_buffer().set_text("")
        self._sources = []
        self._sources_label.set_text("")
        self._sources_revealer.set_reveal_child(False)
        self._streaming_answer = False
        self._answer_revealer.set_reveal_child(True)
        self._spinner.set_visible(True)
        self._spinner.start()

        self._configure_corner_layout()
        self._stack.set_visible_child_name("corner")
        self._corner_panel.add_css_class("corner-enter")

        def _clear_enter_anim() -> bool:
            self._corner_panel.remove_css_class("corner-enter")
            return False

        GLib.timeout_add(320, _clear_enter_anim)

    def toggle(self) -> None:
        if self._mode == "corner" and self.get_visible():
            self._dismiss_corner()
            return
        if self.get_visible():
            self._hide_spotlight()
        else:
            self._mode = "spotlight"
            self.remove_css_class("corner-mode")
            self._configure_spotlight_layout()
            self._stack.set_visible_child_name("spotlight")
            self.entry.set_text("")
            self.entry.set_sensitive(True)
            self._scrim.add_css_class("visible")
            self.set_visible(True)
            self.present()
            self.entry.grab_focus()
            self._refresh_status()

    def _refresh_status(self) -> None:
        try:
            r = httpx.get(f"{DAEMON}/health", timeout=2)
            npu = r.json().get("npu", {})
            state = npu.get("state", "?")
            model = npu.get("model_id") or "no model"
            self.status.set_text(f"{state} · {model}")
        except Exception:  # noqa: BLE001
            self.status.set_text("offline")

    def _set_corner_status(self, message: str) -> None:
        self._corner_status.set_text(message)
        if not self._spinner.get_visible():
            self._spinner.set_visible(True)
            self._spinner.start()

    def _append_corner_token(self, token: str) -> None:
        if not self._streaming_answer:
            self._streaming_answer = True
            self._spinner.stop()
            self._spinner.set_visible(False)
            self._corner_answer.get_buffer().set_text("")
        buf = self._corner_answer.get_buffer()
        end = buf.get_end_iter()
        buf.insert(end, token)

    def _show_corner_answer(self, answer: str, sources: list | None = None) -> None:
        self._spinner.stop()
        self._spinner.set_visible(False)
        self._corner_status.set_text("Done")
        if not self._streaming_answer:
            self._corner_answer.get_buffer().set_text(answer)
        self._answer_revealer.set_reveal_child(True)
        if sources:
            lines = [f"[{s.get('index', '?')}] {s.get('title', s.get('url', ''))}" for s in sources]
            self._sources_label.set_text("\n".join(lines))
            self._sources_revealer.set_reveal_child(True)
        self._busy = False
        self.entry.set_sensitive(True)
        self._refresh_status()

    def _on_submit(self, *_args) -> None:
        text = self.entry.get_text().strip()
        if not text or self._busy:
            return
        self.entry.set_sensitive(False)
        self._enter_corner_mode(text)

        def _work() -> None:
            answer = ""
            sources: list = []
            session_id: str | None = self.session_id
            try:
                with httpx.stream(
                    "POST",
                    f"{DAEMON}/chat/stream",
                    json={"message": text, "session_id": self.session_id},
                    timeout=600,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        etype = event.get("type")
                        if etype == "status":
                            GLib.idle_add(self._set_corner_status, event.get("message", "Thinking…"))
                        elif etype == "research":
                            GLib.idle_add(self._set_corner_status, event.get("message", "Researching…"))
                        elif etype == "tool":
                            msg = event.get("message") or f"Calling {event.get('name', 'tool')}…"
                            GLib.idle_add(self._set_corner_status, msg)
                        elif etype == "token":
                            tok = event.get("text", "")
                            if tok:
                                GLib.idle_add(self._append_corner_token, tok)
                        elif etype == "permission":
                            GLib.idle_add(self._show_permission_dialog, event)
                        elif etype == "sources":
                            sources = event.get("sources", sources)
                        elif etype in ("answer", "done"):
                            answer = event.get("answer") or event.get("text") or answer
                            if event.get("sources"):
                                sources = event["sources"]
                            if event.get("session_id"):
                                session_id = event["session_id"]
                        elif etype == "error":
                            answer = f"Error: {event.get('message', 'unknown error')}"
            except Exception as exc:  # noqa: BLE001
                answer = f"Error: {exc}"

            def _finish() -> bool:
                if session_id:
                    self.session_id = session_id
                self._show_corner_answer(answer or "No response.", sources or None)
                return False

            GLib.idle_add(_finish)

        threading.Thread(target=_work, daemon=True).start()


class KeylaneApp(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id="app.keylane.Spotlight")
        self._toggle_action = Gio.SimpleAction.new("toggle", None)
        self._toggle_action.connect("activate", self._on_toggle_action)
        self.add_action(self._toggle_action)

    def _on_toggle_action(self, *_args) -> None:
        self.activate()

    def do_startup(self) -> None:
        Gtk.Application.do_startup(self)
        apply_spotlight_theme()

    def do_activate(self) -> None:
        win = self.props.active_window
        if not win:
            win = SpotlightWindow(self)
            win.set_application(self)
            self.hold()
        win.toggle()


def _toggle_remote() -> bool:
    try:
        completed = subprocess.run(
            ["gapplication", "action", "app.keylane.Spotlight", "toggle"],
            check=False,
            capture_output=True,
            timeout=2,
        )
        if completed.returncode == 0:
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        with socket.create_connection(("127.0.0.1", 9101), timeout=1) as sock:
            sock.sendall(b"toggle\n")
        return True
    except OSError:
        return False


def _socket_server(app: KeylaneApp) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    for attempt in range(5):
        try:
            server.bind(("127.0.0.1", 9101))
            break
        except OSError:
            time.sleep(0.5)
    else:
        logger.error("could not bind toggle socket on :9101")
        return
    server.listen(5)

    while True:
        conn, _ = server.accept()
        with conn:
            conn.recv(64)

        def _toggle() -> bool:
            app.activate()
            return False

        GLib.idle_add(_toggle)


def _start_ui_daemon() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    env.setdefault("KEYLANE_DATA", str(ROOT / "data"))
    if os.path.isfile(_LAYER_SHELL_LIB) and _LAYER_SHELL_LIB not in env.get("LD_PRELOAD", ""):
        env["LD_PRELOAD"] = _LAYER_SHELL_LIB + (
            f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else ""
        )
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toggle", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.toggle:
        if _toggle_remote():
            return
        _start_ui_daemon()
        for _ in range(30):
            time.sleep(0.1)
            if _toggle_remote():
                return
        print("Keylane UI failed to start.", file=sys.stderr)
        sys.exit(1)

    app = KeylaneApp()
    threading.Thread(target=_socket_server, args=(app,), daemon=True).start()
    app.run(None)


if __name__ == "__main__":
    main()
