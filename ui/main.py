#!/usr/bin/env python3
"""Keylane Spotlight — macOS-style floating command bar."""

from __future__ import annotations

import argparse
import base64
import cairo
import json
import logging
import os
import re
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
gi.require_version("Pango", "1.0")
try:
    gi.require_version("Gtk4LayerShell", "1.0")
    HAS_LAYER_SHELL = True
except ValueError:
    HAS_LAYER_SHELL = False

from gi.repository import Gdk, Gio, GLib, Gtk, Pango  # type: ignore[attr-defined]

if HAS_LAYER_SHELL:
    from gi.repository import Gtk4LayerShell as LayerShell  # type: ignore[attr-defined]

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu.thinking import extract_user_answer, sanitize_response
from ui.clipboard_image import read_image_bytes
from ui.canvas import block_markup, headline_text, is_compact, parse_blocks, plain_text
from ui.canvas import _inline as _inline_markup
from ui.thinking_orb import ThinkingOrb
from ui.settings import FOCUS_GRACE, FOCUS_SETTLE_MS, SettingsWindow
from ui.screenshot import capture_fullscreen, capture_region
from ui.placement import (
    floating_geometry,
    forced_backend,
    move_resize,
    set_always_on_top,
    wayland_session,
    wmctrl_available,
)
from ui.theme import apply_scheme_classes, apply_spotlight_theme, watch_color_scheme, watch_theme
from ui.voice import mic_recording, start_mic, stop_mic

DAEMON = "http://127.0.0.1:9100"
PANEL_WIDTH = 680
CORNER_WIDTH = 380
CORNER_MARGIN = 20
# Cap the answer card so a long reply scrolls instead of running off-screen.
ANSWER_MAX_HEIGHT = 420
THINKING_ORB_SIZE = ThinkingOrb.ORB_SIZE
logger = logging.getLogger(__name__)


def _layer_shell_ok() -> bool:
    """True only on a Wayland display whose compositor speaks wlr-layer-shell."""
    if not HAS_LAYER_SHELL:
        return False
    display = Gdk.Display.get_default()
    # is_supported() asserts GDK_IS_WAYLAND_DISPLAY, so never call it on X11.
    if display is None or "Wayland" not in type(display).__name__:
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


def _monitor_scale() -> int:
    """Logical→device pixel factor. The window manager positions in device px."""
    display = Gdk.Display.get_default()
    if not display:
        return 1
    monitors = display.get_monitors()
    monitor = monitors.get_item(0) if monitors.get_n_items() > 0 else None
    try:
        return max(int(monitor.get_scale_factor()), 1) if monitor else 1
    except (AttributeError, TypeError, ValueError):
        return 1


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


def _configure_layer_shell_thinking(window: Gtk.ApplicationWindow) -> None:
    if not _layer_shell_ok() or not LayerShell.is_layer_window(window):
        return
    LayerShell.set_keyboard_mode(window, LayerShell.KeyboardMode.NONE)
    for edge in (LayerShell.Edge.BOTTOM, LayerShell.Edge.LEFT):
        LayerShell.set_anchor(window, edge, False)
        LayerShell.set_margin(window, edge, 0)
    for edge in (LayerShell.Edge.TOP, LayerShell.Edge.RIGHT):
        LayerShell.set_anchor(window, edge, True)
        LayerShell.set_margin(window, edge, CORNER_MARGIN)


def _configure_layer_shell_corner(window: Gtk.ApplicationWindow) -> None:
    if not _layer_shell_ok() or not LayerShell.is_layer_window(window):
        return
    # ON_DEMAND, not EXCLUSIVE: the answer panel keeps a focusable follow-up
    # entry, but the compositor still lets you type into other windows.
    LayerShell.set_keyboard_mode(window, LayerShell.KeyboardMode.ON_DEMAND)
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
    """No layer shell: use a small floating window sized to its own content.

    The old fullscreen-modal fallback covered the whole screen, which is why the
    HUD stretched edge to edge and why clicks had to be punched through with an
    input region. A window that is only as big as the panel needs neither.
    """
    # Deliberately resizable: a non-resizable window is sized to its natural
    # width, and a wrapped label's natural width is the whole unwrapped string —
    # that is what stretched the HUD across the screen. The window is
    # undecorated, so there is no grip for the user to resize it with anyway.
    window.set_resizable(True)
    window.set_modal(False)
    # A real starting height: a 1px placeholder is what the window keeps if the
    # first measurement lands before the panel has been laid out.
    window.set_default_size(PANEL_WIDTH, 120)


def _floating_geometry(mode: str, width: int, height: int) -> tuple[int, int]:
    return floating_geometry(mode, width, height, _monitor_size(), CORNER_MARGIN)


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
        self._attached_images: list[bytes] = []
        self._image_preview_path: Path | None = None
        self._mic_sync = False
        self._settings_win: SettingsWindow | None = None
        self._input_region_rect: tuple[int, int, int, int] | None = None
        self._region_tick_id = 0
        self._turns: list[tuple[str, str]] = []
        self._floating_size: tuple[int, int, int, int] | None = None
        self._spotlight_had_focus = False
        self._shown_at = 0.0
        self._on_top_applied = False

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
        if not self._layered:
            # A floating window has no fullscreen area to dim, and a scrim that
            # expands would force the window back to full size.
            self._scrim.set_hexpand(False)
            self._scrim.set_vexpand(False)

        self._backdrop = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._backdrop.add_css_class("spotlight-backdrop")
        self._backdrop.set_halign(Gtk.Align.CENTER)
        self._backdrop.set_valign(Gtk.Align.CENTER)
        self._backdrop.set_margin_start(24)
        self._backdrop.set_margin_end(24)
        overlay.add_overlay(self._backdrop)

        self._stack = Gtk.Stack()
        # Not homogeneous: otherwise every page is sized to the widest one and
        # the corner HUD is allocated the full command-bar width — which also
        # widens the click-blocking input region by ~300px.
        self._stack.set_hhomogeneous(False)
        self._stack.set_vhomogeneous(False)
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._stack.set_transition_duration(200)
        self._stack.add_css_class("spotlight-stack")
        self._backdrop.append(self._stack)

        self._build_spotlight_page()
        self._build_thinking_page()
        self._build_corner_page()

        apply_scheme_classes(self)
        watch_color_scheme(lambda _dark: apply_scheme_classes(self))
        # The orbs paint themselves from theme tokens, so a theme change has to
        # reach them by hand — CSS cannot.
        watch_theme(self._redraw_orbs)

        key = Gtk.EventControllerKey.new()
        key.connect("key-released", self._on_key)
        self.add_controller(key)

        if not self._layered:
            # A floating window has no fullscreen scrim to click away on, so
            # losing focus is what dismisses the launcher — the behaviour people
            # expect from a Spotlight-style bar.
            self.connect("notify::is-active", self._on_active_changed)

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

        self.entry = Gtk.Entry()
        self.entry.add_css_class("spotlight-search")
        self.entry.set_placeholder_text("Search or ask Keylane")
        self.entry.set_has_frame(False)
        self.entry.set_hexpand(True)
        self.entry.connect("activate", self._on_submit)
        search_row.append(self.entry)

        self._mic_btn = Gtk.ToggleButton()
        mic_icon = Gtk.Image.new_from_icon_name("audio-input-microphone-symbolic")
        mic_icon.set_pixel_size(18)
        self._mic_btn.set_child(mic_icon)
        self._mic_btn.set_tooltip_text("Voice input — Ctrl+Shift+M")
        self._mic_btn.add_css_class("spotlight-icon-btn")
        self._mic_btn.add_css_class("spotlight-mic-btn")
        self._mic_btn.set_can_focus(False)
        self._mic_btn.connect("toggled", self._on_mic_toggled)
        search_row.append(self._mic_btn)

        shot_btn = Gtk.Button()
        shot_icon = Gtk.Image.new_from_icon_name("camera-photo-symbolic")
        shot_icon.set_pixel_size(18)
        shot_btn.set_child(shot_icon)
        shot_btn.set_tooltip_text("Capture a region to ask about — Ctrl+Shift+S")
        shot_btn.add_css_class("spotlight-icon-btn")
        shot_btn.set_can_focus(False)
        shot_btn.connect("clicked", lambda *_: self._screenshot_region())
        search_row.append(shot_btn)

        self._image_revealer = Gtk.Revealer()
        self._image_revealer.set_reveal_child(False)
        panel.append(self._image_revealer)

        image_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        image_row.add_css_class("spotlight-image-row")
        self._image_revealer.set_child(image_row)

        self._image_preview = Gtk.Image()
        self._image_preview.add_css_class("spotlight-image-preview")
        self._image_preview.set_pixel_size(48)
        image_row.append(self._image_preview)

        self._image_label = Gtk.Label(label="", xalign=0, ellipsize=3)
        self._image_label.add_css_class("spotlight-image-label")
        self._image_label.set_hexpand(True)
        image_row.append(self._image_label)

        clear_img_btn = Gtk.Button()
        clear_img_btn.set_child(Gtk.Image.new_from_icon_name("window-close-symbolic"))
        clear_img_btn.get_child().set_pixel_size(14)  # type: ignore[union-attr]
        clear_img_btn.set_tooltip_text("Remove image")
        clear_img_btn.add_css_class("spotlight-icon-btn")
        clear_img_btn.set_can_focus(False)
        clear_img_btn.connect("clicked", lambda *_: self._clear_attached_image())
        image_row.append(clear_img_btn)

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

    def _build_thinking_page(self) -> None:
        wrap = Gtk.Box()
        wrap.set_halign(Gtk.Align.END)
        wrap.set_valign(Gtk.Align.START)
        wrap.add_css_class("thinking-orb-wrap")
        self._thinking_wrap = wrap
        self._stack.add_named(wrap, "thinking")

        self._thinking_orb = ThinkingOrb()
        self._thinking_orb.set_halign(Gtk.Align.CENTER)
        self._thinking_orb.set_valign(Gtk.Align.CENTER)
        wrap.append(self._thinking_orb)
        self._thinking_orb.connect("map", self._on_thinking_orb_map)

    def _on_thinking_orb_map(self, *_args: object) -> None:
        if self._mode == "thinking":
            GLib.idle_add(self._update_pass_through_input_region)
            GLib.timeout_add(80, self._update_pass_through_input_region)

    def _build_corner_page(self) -> None:
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        panel.add_css_class("corner-panel")
        panel.set_size_request(CORNER_WIDTH, -1)
        panel.set_halign(Gtk.Align.END)
        panel.set_valign(Gtk.Align.START)
        self._corner_panel = panel
        self._stack.add_named(panel, "corner")

        # ── header: indicator + query + dismiss ──────────────────────────
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        header.add_css_class("corner-header")
        panel.append(header)

        # The same orb as the floating one, scaled down — the collapse then
        # reads as one object settling into place rather than a swap.
        self._corner_orb = ThinkingOrb(size=22)
        self._corner_orb.set_valign(Gtk.Align.CENTER)
        self._corner_orb.add_css_class("corner-orb")
        header.append(self._corner_orb)

        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
        title_box.set_hexpand(True)
        header.append(title_box)

        self._corner_title = Gtk.Label(label="Keylane")
        self._corner_title.add_css_class("corner-title")
        self._corner_title.set_halign(Gtk.Align.START)
        self._corner_title.set_ellipsize(3)  # PANGO_ELLIPSIZE_END
        self._corner_title.set_max_width_chars(38)
        title_box.append(self._corner_title)

        self._corner_status = Gtk.Label(label="Thinking…")
        self._corner_status.add_css_class("corner-status")
        self._corner_status.set_halign(Gtk.Align.START)
        self._corner_status.set_ellipsize(3)
        self._corner_status.set_max_width_chars(38)
        title_box.append(self._corner_status)

        close_btn = Gtk.Button()
        close_btn.add_css_class("corner-close")
        close_btn.set_icon_name("window-close-symbolic")
        close_btn.set_tooltip_text("Dismiss (Esc)")
        close_btn.set_valign(Gtk.Align.START)
        close_btn.set_can_focus(False)
        close_btn.connect("clicked", lambda *_: self._dismiss_corner())
        header.append(close_btn)

        # ── answer canvas ────────────────────────────────────────────────
        self._answer_revealer = Gtk.Revealer()
        self._answer_revealer.set_reveal_child(False)
        self._answer_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._answer_revealer.set_transition_duration(240)
        panel.append(self._answer_revealer)

        answer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        answer_box.add_css_class("corner-answer-box")
        self._answer_revealer.set_child(answer_box)

        # A long answer scrolls inside the card instead of growing off-screen.
        self._answer_scroll = Gtk.ScrolledWindow()
        self._answer_scroll.add_css_class("corner-scroll")
        # NEVER so the canvas wraps to the viewport width rather than scrolling
        # sideways. It reports the child's natural width upward, which is why the
        # floating window sets its own size instead of hugging natural size.
        self._answer_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._answer_scroll.set_propagate_natural_height(True)
        self._answer_scroll.set_propagate_natural_width(False)
        self._answer_scroll.set_max_content_height(ANSWER_MAX_HEIGHT)
        answer_box.append(self._answer_scroll)

        self._canvas_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=9)
        self._canvas_card.add_css_class("corner-canvas-card")
        self._answer_scroll.set_child(self._canvas_card)

        self._details_revealer = Gtk.Revealer()
        self._details_revealer.set_reveal_child(False)
        self._details_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._details_revealer.set_transition_duration(180)
        answer_box.append(self._details_revealer)

        details_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._details_revealer.set_child(details_row)

        self._details_btn = Gtk.Button(label="More details")
        self._details_btn.add_css_class("corner-details-btn")
        self._details_btn.set_can_focus(False)
        self._details_btn.connect("clicked", self._on_toggle_details)
        details_row.append(self._details_btn)

        copy_btn = Gtk.Button(label="Copy")
        copy_btn.add_css_class("corner-details-btn")
        copy_btn.set_can_focus(False)
        copy_btn.connect("clicked", lambda *_: self._copy_answer())
        details_row.append(copy_btn)

        self._canvas_full_answer = ""
        self._canvas_summary = ""
        self._canvas_showing_full = False

        # ── follow-up ────────────────────────────────────────────────────
        self._followup_revealer = Gtk.Revealer()
        self._followup_revealer.set_reveal_child(False)
        self._followup_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._followup_revealer.set_transition_duration(200)
        panel.append(self._followup_revealer)

        followup_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        followup_row.add_css_class("corner-followup-row")
        self._followup_revealer.set_child(followup_row)

        self._followup_entry = Gtk.Entry()
        self._followup_entry.add_css_class("corner-followup")
        self._followup_entry.set_placeholder_text("Ask a follow-up…")
        self._followup_entry.set_has_frame(False)
        self._followup_entry.set_hexpand(True)
        self._followup_entry.connect("activate", self._on_followup)
        followup_row.append(self._followup_entry)

        send_btn = Gtk.Button()
        send_btn.set_child(Gtk.Image.new_from_icon_name("go-up-symbolic"))
        send_btn.add_css_class("corner-send-btn")
        send_btn.set_tooltip_text("Send")
        send_btn.set_can_focus(False)
        send_btn.connect("clicked", self._on_followup)
        followup_row.append(send_btn)

    def _copy_answer(self) -> None:
        text = plain_text(self._canvas_full_answer or self._canvas_summary)
        if not text:
            return
        display = Gdk.Display.get_default()
        if display is not None:
            display.get_clipboard().set(text)
            self._set_corner_status("Copied to clipboard")

    def _on_followup(self, *_args: object) -> None:
        text = self._followup_entry.get_text().strip()
        if not text or self._busy:
            return
        self._followup_entry.set_text("")
        self._submit_query(text)

    def _on_active_changed(self, *_args: object) -> None:
        """Dismiss the launcher when it loses focus — carefully.

        Focus around a freshly mapped, skip-taskbar window bounces: the manager
        may hand it focus and take it straight back. Dismissing on the first
        inactive edge closes the launcher the instant it opens, so this waits
        for focus to be genuinely gone and ignores a short grace period after
        the window appears.
        """
        if self.get_property("is-active"):
            self._spotlight_had_focus = True
            return
        if not self._spotlight_had_focus:
            return
        if time.monotonic() - self._shown_at < FOCUS_GRACE:
            return

        def _confirm() -> bool:
            if (
                not self.get_property("is-active")
                and self._mode == "spotlight"
                and self.get_visible()
                and not self._busy
                and not mic_recording()
                and not (self._settings_win is not None and self._settings_win.get_visible())
            ):
                self._hide_spotlight()
            return False

        GLib.timeout_add(FOCUS_SETTLE_MS, _confirm)

    def _on_scrim_click(self, _gesture, _n_press: int, _x: float, _y: float) -> None:
        if self._mode == "spotlight" and not self._busy:
            self._hide_spotlight()

    def _on_key(self, _ctrl: Gtk.EventControllerKey, keyval: int, _keycode: int, _state: Gdk.ModifierType) -> bool:
        if keyval == Gdk.KEY_Escape:
            if mic_recording():
                self._set_mic_active(False)
                stop_mic(on_done=self._mic_transcribed, on_error=lambda msg: GLib.idle_add(self._show_toast, msg[:40]))
            if self._mode in ("corner", "thinking"):
                self._dismiss_corner()
            else:
                self._hide_spotlight()
            return True
        if keyval in (Gdk.KEY_comma, Gdk.KEY_semicolon) and (_state & Gdk.ModifierType.CONTROL_MASK):
            self._open_settings()
            return True
        if keyval == Gdk.KEY_m and (_state & Gdk.ModifierType.CONTROL_MASK) and (
            _state & Gdk.ModifierType.SHIFT_MASK
        ):
            self.toggle_mic()
            return True
        if keyval == Gdk.KEY_v and (_state & Gdk.ModifierType.CONTROL_MASK) and (
            _state & Gdk.ModifierType.SHIFT_MASK
        ):
            self._paste_image()
            return True
        if keyval == Gdk.KEY_s and (_state & Gdk.ModifierType.CONTROL_MASK) and (
            _state & Gdk.ModifierType.SHIFT_MASK
        ):
            self._screenshot_region()
            return True
        if keyval == Gdk.KEY_S and (_state & Gdk.ModifierType.CONTROL_MASK) and (
            _state & Gdk.ModifierType.SHIFT_MASK
        ):
            self._screenshot_full()
            return True
        return False

    def _redraw_orbs(self) -> None:
        for orb in (self._thinking_orb, self._corner_orb):
            orb.queue_draw()

    def _show_toast(self, message: str) -> None:
        self.status.set_text(message[:40])

    def _open_settings(self) -> None:
        try:
            if self._settings_win is None:
                self._settings_win = SettingsWindow(self, independent=self._layered)
                self._settings_win.set_toast_callback(self._show_toast)
                self._settings_win.set_scheme_callback(lambda: apply_scheme_classes(self))
                self._settings_win.set_dismiss_callback(self._on_settings_dismissed)
            # Show the window first, then fill it. Loading first meant the
            # click on the gear did nothing visible until the daemon answered.
            self._settings_win.present_centered()
            self._settings_win.load_settings()
        except Exception:  # noqa: BLE001
            logger.exception("failed to open settings")
            self._show_toast("Settings failed to open")

    def _on_settings_dismissed(self) -> None:
        """A click elsewhere closed settings — take the launcher with it.

        Unless the click landed on the launcher itself, which is then the
        window the user is working in.
        """
        if self._mode == "spotlight" and self.get_visible() and not self.get_property("is-active"):
            self._hide_spotlight()

    def _set_mic_active(self, active: bool) -> None:
        self._mic_sync = True
        self._mic_btn.set_active(active)
        if active:
            self._mic_btn.add_css_class("spotlight-mic-active")
        else:
            self._mic_btn.remove_css_class("spotlight-mic-active")
        self._mic_sync = False

    def toggle_mic(self) -> None:
        if self._busy:
            return
        if not self.get_visible() or self._mode != "spotlight":
            self._mode = "spotlight"
            self.remove_css_class("corner-mode")
            self.remove_css_class("thinking-mode")
            self._configure_spotlight_layout()
            self._stack.set_visible_child_name("spotlight")
            self._spotlight_had_focus = False
            self._shown_at = time.monotonic()
            if self._layered:
                self._scrim.add_css_class("visible")
            self.set_visible(True)
            self.present()
            self._start_region_tracking()
            self.entry.grab_focus()

        if mic_recording():
            self._set_mic_active(False)
            stop_mic(
                on_done=self._mic_transcribed,
                on_error=lambda msg: GLib.idle_add(self._show_toast, msg[:40]),
            )
        else:
            self._set_mic_active(True)
            self._show_toast("Listening…")
            start_mic()

    def _on_mic_toggled(self, btn: Gtk.ToggleButton) -> None:
        if self._mic_sync:
            return
        if btn.get_active():
            if self._busy:
                self._set_mic_active(False)
                return
            start_mic()
            btn.add_css_class("spotlight-mic-active")
            self._show_toast("Listening…")
        else:
            if mic_recording():
                stop_mic(
                    on_done=self._mic_transcribed,
                    on_error=lambda msg: GLib.idle_add(self._show_toast, msg[:40]),
                )
            btn.remove_css_class("spotlight-mic-active")

    def _mic_transcribed(self, text: str) -> None:
        def _ui() -> bool:
            self._set_mic_active(False)
            if text:
                self.entry.set_text(text)
                self._show_toast("Transcribed")
            elif mic_recording():
                pass
            else:
                self._show_toast("No speech detected")
            return False

        GLib.idle_add(_ui)

    def _clear_attached_image(self) -> None:
        self._attached_images.clear()
        if self._image_preview_path and self._image_preview_path.is_file():
            try:
                self._image_preview_path.unlink()
            except OSError:
                pass
        self._image_preview_path = None
        self._image_revealer.set_reveal_child(False)

    def _show_attached_image(self, data: bytes, label: str) -> None:
        self._clear_attached_image()
        self._attached_images = [data]
        preview = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / f"keylane-attach-{time.time_ns()}.png"
        preview.write_bytes(data)
        self._image_preview_path = preview
        self._image_preview.set_from_file(str(preview))
        self._image_label.set_text(label)
        self._image_revealer.set_reveal_child(True)
        self._show_toast("Image attached")

    def _attach_image_file(self, path: Path, label: str | None = None) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self._show_toast("Could not read image")
            return
        self._show_attached_image(data, label or path.name)

    def _paste_image(self) -> None:
        if self._busy:
            return

        def on_ready(data: bytes | None) -> None:
            if data:
                self._show_attached_image(data, "Pasted image")
            else:
                self._show_toast("No image in clipboard")

        read_image_bytes(on_ready)

    def _before_screenshot_capture(self) -> None:
        self._capture_restore_visible = self.get_visible()
        self._scrim.remove_css_class("visible")
        self.set_visible(False)

    def _after_screenshot_capture(self, path: Path | None, label: str) -> None:
        if getattr(self, "_capture_restore_visible", False):
            self.set_visible(True)
            if self._mode == "spotlight":
                self._scrim.add_css_class("visible")
            self.present()
            self.entry.grab_focus()
        if path:
            self._attach_image_file(path, label)
            try:
                path.unlink()
            except OSError:
                pass
        else:
            self._show_toast("Screenshot cancelled")

    def _screenshot_region(self) -> None:
        if self._busy:
            return
        self._before_screenshot_capture()

        def _work() -> None:
            time.sleep(0.2)
            shot = capture_region()
            GLib.idle_add(self._after_screenshot_capture, shot, "Screenshot region")

        threading.Thread(target=_work, daemon=True).start()

    def _screenshot_full(self) -> None:
        if self._busy:
            return
        self._before_screenshot_capture()

        def _work() -> None:
            time.sleep(0.2)
            shot = capture_fullscreen()
            GLib.idle_add(self._after_screenshot_capture, shot, "Screenshot")

        threading.Thread(target=_work, daemon=True).start()

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
        self._stop_region_tracking()
        self._scrim.remove_css_class("visible")
        self.set_visible(False)

    def _dismiss_corner(self) -> None:
        self._busy = False
        self._stop_region_tracking()
        self._turns.clear()
        self.remove_css_class("corner-mode")
        self.remove_css_class("corner-dark")
        self.remove_css_class("thinking-mode")
        self._scrim.remove_css_class("visible")
        self._configure_spotlight_layout()
        self._stack.set_visible_child_name("spotlight")
        self._mode = "spotlight"
        self.entry.set_sensitive(True)
        self.set_visible(False)
        self._clear_input_region()

    def _clear_input_region(self) -> None:
        self._input_region_rect = None
        if not self.get_realized():
            return
        surface = self.get_surface()
        if surface is not None:
            surface.set_input_region(None)

    def _widget_surface_rect(self, widget: Gtk.Widget) -> tuple[int, int, int, int] | None:
        if not widget.get_realized():
            return None
        native = widget.get_native()
        if native is None:
            return None
        origin = widget.translate_coordinates(native, 0, 0)
        if origin is None:
            return None
        x, y = origin
        alloc = widget.get_allocation()
        return int(x), int(y), alloc.width, alloc.height

    def _interactive_widget(self) -> Gtk.Widget | None:
        """The only widget that should catch clicks in overlay modes."""
        if self._mode == "thinking":
            return self._thinking_orb
        if self._mode == "corner":
            return self._corner_panel
        return None

    def _update_pass_through_input_region(self) -> bool:
        """Shrink the input region to the visible panel.

        The window is a fullscreen overlay surface. Without this, every click
        anywhere on screen lands on Keylane instead of the window underneath —
        so while the orb or the answer card is up, the rest of the desktop is
        unreachable. Restricting the input region to the panel's own rectangle
        keeps it drawn above everything while the empty space stays clickable.
        """
        if not self._layered:
            # The floating window is exactly the panel, so it needs no hole
            # punched in it. Setting one here left a 56x56 orb-sized patch as
            # the only clickable part of the answer HUD.
            self._clear_input_region()
            return False
        if not self.get_realized():
            return False
        surface = self.get_surface()
        if surface is None:
            return False

        widget = self._interactive_widget()
        if widget is None:
            if self._input_region_rect is not None:
                surface.set_input_region(None)
                self._input_region_rect = None
            return False

        rect = self._widget_surface_rect(widget)
        if rect is None or rect[2] <= 0 or rect[3] <= 0:
            return False
        if rect == self._input_region_rect:
            return False

        x, y, w, h = rect
        # A little slack so the panel's shadow and rounded edges stay grabbable.
        pad = 6
        surface.set_input_region(
            cairo.Region(cairo.RectangleInt(max(x - pad, 0), max(y - pad, 0), w + pad * 2, h + pad * 2))
        )
        self._input_region_rect = rect
        return False

    def _clear_backdrop_insets(self) -> None:
        """A floating window is the panel; margins would only pad the window."""
        for setter in (
            self._backdrop.set_margin_top,
            self._backdrop.set_margin_bottom,
            self._backdrop.set_margin_start,
            self._backdrop.set_margin_end,
        ):
            setter(0)

    def _place_floating(self, mode: str) -> None:
        """Size and position the window when there is no layer shell.

        Deferred to idle: the panel has not been measured yet at the moment the
        mode flips, so asking for its size here would return the previous one.
        """
        if self._layered:
            return
        self._clear_backdrop_insets()
        self._floating_size = None
        GLib.idle_add(self._apply_floating_geometry, mode)

    def _apply_floating_geometry(self, mode: str) -> bool:
        if self._layered or self._mode != mode or not self.get_realized():
            return False

        width = PANEL_WIDTH if mode == "spotlight" else (
            THINKING_ORB_SIZE if mode == "thinking" else CORNER_WIDTH
        )
        child = {
            "spotlight": self._spotlight_panel,
            "thinking": self._thinking_orb,
            "corner": self._corner_panel,
        }[mode]
        natural_height = child.measure(Gtk.Orientation.VERTICAL, width)[1]
        height = max(natural_height, THINKING_ORB_SIZE if mode == "thinking" else 1)

        # GTK works in logical pixels; the window manager takes device pixels.
        # Both the position *and* the size need the scale factor — an unscaled
        # size on a 2x display gives a half-size window that clips the panel.
        scale = _monitor_scale()
        x, y = (v * scale for v in _floating_geometry(mode, width, height))
        device = (x, y, width * scale, height * scale)
        if device == self._floating_size:
            return False

        self.set_default_size(width, height)
        self._ensure_on_top()
        # Cache only on success: the first attempt usually lands before the
        # window manager has the window in its client list, and caching a failed
        # move would leave the panel wherever the WM happened to put it.
        if move_resize(self, *device):
            self._floating_size = device
        return False

    def _ensure_on_top(self) -> None:
        """Re-assert always-on-top; the hint is lost when the window is remapped."""
        if self._layered:
            return
        if set_always_on_top(self):
            self._on_top_applied = True

    def _track_floating_size(self) -> None:
        """Follow the panel as it grows (answer lands, details expand)."""
        if self._layered:
            return
        self._apply_floating_geometry(self._mode)

    def _start_region_tracking(self) -> None:
        """Keep the window's geometry in step with its content, every frame.

        GTK4 has no external size-allocate signal, and the panel changes size
        whenever an answer lands or 'More details' expands. The tick early-outs
        on unchanged geometry, so it is cheap. It also covers the first layout
        pass: a panel measured before it has been laid out reports a placeholder
        size, and without a retry the window keeps it — which is what collapsed
        the spotlight into a sliver.
        """
        if self._region_tick_id:
            return

        def _tick(_widget: Gtk.Widget, _clock: object) -> bool:
            if not self.get_visible():
                self._region_tick_id = 0
                return GLib.SOURCE_REMOVE
            if self._layered:
                if self._mode not in ("thinking", "corner"):
                    self._region_tick_id = 0
                    return GLib.SOURCE_REMOVE
                self._update_pass_through_input_region()
            else:
                self._track_floating_size()
            return GLib.SOURCE_CONTINUE

        self._region_tick_id = self.add_tick_callback(_tick)

    def _stop_region_tracking(self) -> None:
        if self._region_tick_id:
            self.remove_tick_callback(self._region_tick_id)
            self._region_tick_id = 0

    def _configure_spotlight_layout(self) -> None:
        self._backdrop.set_halign(Gtk.Align.CENTER)
        self._backdrop.set_valign(Gtk.Align.CENTER)
        self._backdrop.set_margin_top(0)
        self._backdrop.set_margin_bottom(0)
        self._backdrop.set_margin_start(24)
        self._backdrop.set_margin_end(24)
        if self._layered:
            _configure_layer_shell_spotlight(self)
        else:
            self._place_floating("spotlight")

    def _configure_thinking_layout(self) -> None:
        self._backdrop.set_halign(Gtk.Align.END)
        self._backdrop.set_valign(Gtk.Align.START)
        self._backdrop.set_margin_top(CORNER_MARGIN)
        self._backdrop.set_margin_bottom(0)
        self._backdrop.set_margin_start(0)
        self._backdrop.set_margin_end(CORNER_MARGIN)
        if self._layered:
            _configure_layer_shell_thinking(self)
        else:
            self._place_floating("thinking")

    def _configure_corner_layout(self) -> None:
        self._backdrop.set_halign(Gtk.Align.END)
        self._backdrop.set_valign(Gtk.Align.START)
        self._backdrop.set_margin_top(CORNER_MARGIN)
        self._backdrop.set_margin_bottom(0)
        self._backdrop.set_margin_start(0)
        self._backdrop.set_margin_end(CORNER_MARGIN)
        if self._layered:
            _configure_layer_shell_corner(self)
        else:
            self._place_floating("corner")

    def _promote_to_corner_panel(self) -> None:
        if self._stack.get_visible_child_name() == "corner":
            return
        self.remove_css_class("thinking-mode")
        self._mode = "corner"
        self.add_css_class("corner-dark")
        self._configure_corner_layout()
        self._clear_input_region()
        self._start_region_tracking()
        self._stack.set_visible_child_name("corner")
        self._corner_panel.add_css_class("corner-enter")
        self._answer_revealer.set_reveal_child(True)
        self._corner_orb.set_state("thinking")

        def _clear_enter_anim() -> bool:
            self._corner_panel.remove_css_class("corner-enter")
            return False

        GLib.timeout_add(320, _clear_enter_anim)

    def _enter_corner_mode(self, query: str) -> None:
        self._mode = "thinking"
        self._busy = True
        self.add_css_class("corner-mode")
        self.add_css_class("corner-dark")
        self.add_css_class("thinking-mode")
        self._scrim.remove_css_class("visible")

        self._corner_title.set_text(query)
        self._corner_status.set_text("Thinking…")
        self._clear_canvas()
        self._canvas_full_answer = ""
        self._canvas_summary = ""
        self._canvas_showing_full = False
        self._details_revealer.set_reveal_child(False)
        self._details_btn.set_label("More details")
        self._sources = []
        self._streaming_answer = False
        self._answer_revealer.set_reveal_child(False)
        self._followup_revealer.set_reveal_child(False)
        self._thinking_orb.set_state("thinking")
        self._corner_orb.set_state("thinking")
        self._corner_panel.remove_css_class("settled")
        self._thinking_orb.set_tooltip_text("Keylane is thinking…")

        self._configure_thinking_layout()
        self._stack.set_visible_child_name("thinking")
        self.set_visible(True)
        self.present()
        if self._layered:
            GLib.idle_add(self._update_pass_through_input_region)
        self._start_region_tracking()

    def toggle(self) -> None:
        if self._mode in ("corner", "thinking") and self.get_visible():
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
            self._clear_attached_image()
            self._spotlight_had_focus = False
            self._shown_at = time.monotonic()
            if self._layered:
                self._scrim.add_css_class("visible")
            self.set_visible(True)
            self.present()
            self._start_region_tracking()
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
        text = message or "Thinking…"
        self._thinking_orb.set_tooltip_text(text)
        if self._mode != "corner":
            return
        self._corner_status.set_text(text)

    def _sanitize_display_text(self, text: str) -> str:
        return extract_user_answer(text)

    def _clear_canvas(self) -> None:
        child = self._canvas_card.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._canvas_card.remove(child)
            child = nxt

    def _canvas_label(
        self,
        markup: str,
        css: str,
        *,
        selectable: bool = True,
        wrap: bool = True,
    ) -> Gtk.Label:
        label = Gtk.Label()
        label.add_css_class(css)
        # FILL, not START: a START-aligned label is allocated its *natural*
        # width, and Pango's natural width for a hyphenating label is narrow —
        # which is why headings came out as "HIGHLIGHT-S". xalign keeps the
        # text left-aligned inside the full-width allocation.
        label.set_halign(Gtk.Align.FILL)
        label.set_xalign(0)
        label.set_wrap(wrap)
        if wrap:
            # WORD, never WORD_CHAR: WORD_CHAR makes GtkLabel report its
            # minimum width as the *unwrapped* text width, which drags the
            # whole HUD wide.
            label.set_wrap_mode(Gtk.WrapMode.WORD)
        else:
            label.set_ellipsize(Pango.EllipsizeMode.END)
        label.set_selectable(selectable)
        try:
            label.set_markup(markup)
        except Exception:  # noqa: BLE001 — fall back to literal text on bad markup
            label.set_text(re.sub(r"<[^>]+>", "", markup))
        return label

    def _render_canvas(self, answer: str) -> None:
        """Build widgets from the parsed blocks — a document, not one paragraph."""
        self._clear_canvas()
        blocks = parse_blocks(answer)
        if not blocks:
            # Last resort so the card is never blank: show whatever came back,
            # or say plainly that there was nothing in it.
            text = answer.strip() or "No answer text was returned."
            self._canvas_card.append(
                self._canvas_label(GLib.markup_escape_text(text), "canvas-text")
            )
            return

        for block in blocks:
            if block.kind == "rule":
                sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
                sep.add_css_class("canvas-rule")
                self._canvas_card.append(sep)

            elif block.kind in ("headline", "heading", "text", "quote"):
                css = {
                    "headline": "canvas-headline",
                    "heading": "canvas-heading",
                    "text": "canvas-text",
                    "quote": "canvas-quote",
                }[block.kind]
                self._canvas_card.append(
                    self._canvas_label(block_markup(block), css, wrap=block.kind != "heading")
                )

            elif block.kind in ("bullets", "numbers"):
                listbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
                listbox.add_css_class("canvas-list")
                for index, item in enumerate(block.items, start=1):
                    row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
                    marker = Gtk.Label(label="•" if block.kind == "bullets" else f"{index}.")
                    marker.add_css_class("canvas-marker")
                    marker.set_valign(Gtk.Align.START)
                    marker.set_xalign(0)
                    marker.set_halign(Gtk.Align.START)
                    row.append(marker)
                    body = self._canvas_label(_inline_markup(item), "canvas-text")
                    body.set_hexpand(True)
                    row.append(body)
                    listbox.append(row)
                self._canvas_card.append(listbox)

            elif block.kind == "kv":
                grid = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
                grid.add_css_class("canvas-kv")
                for key, value in block.pairs:
                    cell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
                    cell.add_css_class("canvas-kv-row")
                    cell.append(self._canvas_label(_inline_markup(key), "canvas-kv-key", wrap=False))
                    cell.append(self._canvas_label(_inline_markup(value), "canvas-kv-value"))
                    grid.append(cell)
                self._canvas_card.append(grid)

            elif block.kind == "code":
                scroller = Gtk.ScrolledWindow()
                scroller.add_css_class("canvas-code-scroll")
                scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
                scroller.set_propagate_natural_width(False)
                code = Gtk.Label(label=block.text)
                code.add_css_class("canvas-code")
                code.set_halign(Gtk.Align.START)
                code.set_xalign(0)
                code.set_selectable(True)
                scroller.set_child(code)
                self._canvas_card.append(scroller)

        self._canvas_card.queue_resize()
        self._corner_panel.queue_resize()

    def _set_corner_answer_text(self, answer: str, sources: list | None = None) -> None:
        """Sources are intentionally not shown; the canvas is the whole answer."""
        body = plain_text(answer) or answer.strip()
        self._canvas_full_answer = body
        self._canvas_summary = headline_text(body) or body
        # Show the whole thing only when it is genuinely short. Research
        # excerpts run to hundreds of characters each and filled the card.
        self._canvas_showing_full = is_compact(body)

        self._render_canvas(body if self._canvas_showing_full else self._canvas_summary)
        self._details_revealer.set_reveal_child(not self._canvas_showing_full)
        self._details_btn.set_label("Show less" if self._canvas_showing_full else "More details")

    def _on_toggle_details(self, *_args: object) -> None:
        if not self._canvas_full_answer:
            return
        if self._canvas_showing_full:
            self._canvas_showing_full = False
            self._render_canvas(self._canvas_summary)
            self._details_btn.set_label("More details")
        else:
            self._canvas_showing_full = True
            self._render_canvas(self._canvas_full_answer)
            self._details_btn.set_label("Show less")
        self._resize_corner_panel()

    def _resize_corner_panel(self) -> None:
        self._corner_panel.queue_resize()
        self.queue_resize()

    def _replace_corner_answer(self, answer: str) -> None:
        if self._mode == "thinking":
            self._promote_to_corner_panel()
        self._corner_orb.set_state("done")
        self._set_corner_answer_text(answer)
        self._resize_corner_panel()
        self._streaming_answer = True
        self._answer_revealer.set_reveal_child(True)

    def _show_corner_answer(self, answer: str, sources: list | None = None) -> None:
        if self._mode == "thinking":
            self._promote_to_corner_panel()
        self._corner_orb.set_state("done")
        self._corner_panel.add_css_class("settled")
        self._corner_status.set_text(self._answer_footnote(sources))
        self._set_corner_answer_text(answer, sources)
        self._resize_corner_panel()
        self._streaming_answer = False
        self._answer_revealer.set_reveal_child(True)
        self._followup_revealer.set_reveal_child(True)
        self._busy = False
        self.entry.set_sensitive(True)
        self._refresh_status()

    def _answer_footnote(self, sources: list | None) -> str:
        count = len(sources or [])
        if count == 1:
            return "Answered · 1 source"
        if count:
            return f"Answered · {count} sources"
        return "Answered"

    def _on_submit(self, *_args) -> None:
        text = self.entry.get_text().strip()
        images = list(self._attached_images)
        if not text and not images:
            return
        self.entry.set_sensitive(False)
        self._submit_query(text, images)

    def _submit_query(self, text: str, images: list[bytes] | None = None) -> None:
        """Single entry point for both the spotlight bar and the follow-up field."""
        images = images or []
        if (not text and not images) or self._busy:
            return

        display_query = text or "Describe this image"
        self._enter_corner_mode(display_query)
        encoded_images = [base64.b64encode(img).decode("ascii") for img in images]
        self._clear_attached_image()

        def _work() -> None:
            answer = ""
            sources: list = []
            session_id: str | None = self.session_id
            try:
                with httpx.stream(
                    "POST",
                    f"{DAEMON}/chat/stream",
                    json={
                        "message": text or "Describe this image.",
                        "session_id": self.session_id,
                        "images": encoded_images,
                    },
                    timeout=600,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        etype = event.get("type")
                        if etype in ("status", "research"):
                            GLib.idle_add(self._set_corner_status, event.get("message", "Thinking…"))
                        elif etype == "tool":
                            msg = event.get("message") or f"Calling {event.get('name', 'tool')}…"
                            GLib.idle_add(self._set_corner_status, msg)
                        elif etype == "replace_answer":
                            GLib.idle_add(self._replace_corner_answer, event.get("text", ""))
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
                resolved = answer or "No response."
                self._turns.append((text, resolved))
                self._show_corner_answer(resolved, sources or None)
                return False

            GLib.idle_add(_finish)

        threading.Thread(target=_work, daemon=True).start()


class KeylaneApp(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id="app.keylane.Spotlight")
        self._started = False
        self._toggle_action = Gio.SimpleAction.new("toggle", None)
        self._toggle_action.connect("activate", self._on_toggle_action)
        self.add_action(self._toggle_action)
        self._mic_action = Gio.SimpleAction.new("mic", None)
        self._mic_action.connect("activate", self._on_mic_action)
        self.add_action(self._mic_action)

    def ensure_window(self) -> SpotlightWindow:
        win = self.props.active_window
        if win is None:
            win = SpotlightWindow(self)
            win.set_application(self)
            self.hold()  # the UI is a resident service, not a one-shot window
        return win  # type: ignore[return-value]

    def toggle_window(self) -> None:
        self.ensure_window().toggle()

    def _on_toggle_action(self, *_args) -> None:
        self.toggle_window()

    def _on_mic_action(self, *_args) -> None:
        self.ensure_window().toggle_mic()

    def do_startup(self) -> None:
        Gtk.Application.do_startup(self)
        apply_spotlight_theme()

    def do_activate(self) -> None:
        """GTK emits activate once on run(); that must not pop the launcher.

        The UI runs as a login service, so the first activation only builds the
        window and leaves it hidden. Later activations are real user requests.
        """
        self.ensure_window()
        if not self._started:
            self._started = True
            return
        self.toggle_window()


def _remote_command(command: str) -> bool:
    try:
        completed = subprocess.run(
            ["gapplication", "action", "app.keylane.Spotlight", command],
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
            sock.sendall(f"{command}\n".encode())
        return True
    except OSError:
        return False


def _toggle_remote() -> bool:
    return _remote_command("toggle")


def _mic_remote() -> bool:
    return _remote_command("mic")


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
            raw = conn.recv(64).decode("utf-8", errors="ignore").strip() or "toggle"
        cmd = raw.split("\n", 1)[0].strip().lower()

        def _dispatch(app_ref: KeylaneApp = app, command: str = cmd) -> bool:
            if command == "mic":
                app_ref.ensure_window().toggle_mic()
            else:
                app_ref.toggle_window()
            return False

        GLib.idle_add(_dispatch)


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


def _select_backend() -> None:
    """Fall back to XWayland when the compositor has no wlr-layer-shell.

    GNOME/Mutter is the common case. A plain Wayland toplevel cannot stay on
    top of other windows or position itself, so neither the floating orb nor the
    answer HUD can behave correctly there. Under X11 both are possible, so we
    re-exec onto the X11 backend once, before any window is created.
    """
    if os.environ.get("KEYLANE_BACKEND_PRIMED") == "1":
        return
    forced = forced_backend()
    if forced == "layer" or os.environ.get("GDK_BACKEND"):
        return
    if not wayland_session():
        return

    use_x11 = forced == "x11"
    if not use_x11:
        try:
            Gtk.init()
            use_x11 = not _layer_shell_ok()
        except Exception:  # noqa: BLE001
            use_x11 = True

    if not use_x11:
        return
    if not wmctrl_available():
        logger.warning(
            "No wlr-layer-shell and wmctrl is missing — the overlay cannot be kept "
            "on top. Install wmctrl (sudo dnf install wmctrl)."
        )
    logger.info("compositor has no layer shell; restarting on the X11 backend")
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["KEYLANE_BACKEND_PRIMED"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toggle", action="store_true")
    parser.add_argument("--mic", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mic:
        if _mic_remote():
            return
        _start_ui_daemon()
        for _ in range(30):
            time.sleep(0.1)
            if _mic_remote():
                return
        print("Keylane UI failed to start.", file=sys.stderr)
        sys.exit(1)

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

    _select_backend()
    app = KeylaneApp()
    threading.Thread(target=_socket_server, args=(app,), daemon=True).start()
    app.run(None)


if __name__ == "__main__":
    main()
