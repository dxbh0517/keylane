"""GTK4 / libadwaita Super-key launcher UI."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, GLib, Gtk  # noqa: E402

import httpx

logger = logging.getLogger(__name__)

GATEWAY = "http://127.0.0.1:9100"

_theme_provider: Gtk.CssProvider | None = None


def _apply_launcher_theme() -> None:
    """Load active theme CSS from the gateway, then local active/default files."""
    global _theme_provider
    css_text = ""
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{GATEWAY}/api/themes/active/launcher.css")
            if response.status_code == 200 and response.text.strip():
                css_text = response.text
    except Exception:  # noqa: BLE001
        pass
    if not css_text:
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        home_root = Path.home() / ".local" / "share" / "ai-gateway"
        for candidate in (
            home_root / "themes" / "active-launcher.css",
            root / "themes" / "active-launcher.css",
            home_root / "themes" / "default" / "launcher.css",
            root / "themes" / "default" / "launcher.css",
        ):
            path = candidate.resolve()
            if path.exists():
                css_text = path.read_text(encoding="utf-8")
                break
    if not css_text:
        return
    try:
        display = Gdk.Display.get_default()
        if display is None:
            return
        if _theme_provider is not None:
            Gtk.StyleContext.remove_provider_for_display(display, _theme_provider)
        provider = Gtk.CssProvider()
        provider.load_from_data(css_text.encode("utf-8"))
        Gtk.StyleContext.add_provider_for_display(
            display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 10
        )
        _theme_provider = provider
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to apply launcher theme: %s", exc)


class LauncherWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="Keylane")
        self.set_default_size(560, 220)
        self.set_resizable(True)
        self.add_css_class("ag-shell")
        self._projects: list[dict[str, str]] = []
        self._status: dict[str, bool] = {}
        self._pending_task_id: str | None = None
        self._state = "IDLE"

        _apply_launcher_theme()
        self._build()
        self.connect("map", self._on_map)
        GLib.timeout_add_seconds(5, self._poll_status)

    def _build(self) -> None:
        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_title_widget(Gtk.Label(label="Keylane"))
        toolbar.add_top_bar(header)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        box.set_margin_start(16)
        box.set_margin_end(16)

        # Input row
        input_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.entry = Gtk.Entry()
        self.entry.set_placeholder_text("Ask your computer…")
        self.entry.set_hexpand(True)
        self.entry.connect("activate", self._on_send)
        input_row.append(self.entry)

        self.mic_btn = Gtk.Button(label="Mic")
        self.mic_btn.set_tooltip_text("Push-to-talk (records briefly)")
        self.mic_btn.connect("clicked", self._on_mic)
        input_row.append(self.mic_btn)
        box.append(input_row)

        # Project + status
        meta = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.project_dropdown = Gtk.DropDown()
        self.project_dropdown.set_hexpand(True)
        meta.append(Gtk.Label(label="Project:"))
        meta.append(self.project_dropdown)

        self.local_only = Gtk.CheckButton(label="Local Only")
        meta.append(self.local_only)
        box.append(meta)

        self.status_label = Gtk.Label(label="Workers: …")
        self.status_label.set_xalign(0.0)
        self.status_label.add_css_class("dim-label")
        box.append(self.status_label)

        self.progress = Gtk.Label(label="")
        self.progress.set_xalign(0.0)
        self.progress.set_wrap(True)
        box.append(self.progress)

        # Actions
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        send_btn = Gtk.Button(label="Send")
        send_btn.add_css_class("suggested-action")
        send_btn.connect("clicked", self._on_send)
        actions.append(send_btn)

        send_hide = Gtk.Button(label="Send & Hide")
        send_hide.connect("clicked", self._on_send_hide)
        actions.append(send_hide)

        self.allow_btn = Gtk.Button(label="Allow")
        self.allow_btn.add_css_class("suggested-action")
        self.allow_btn.set_sensitive(False)
        self.allow_btn.connect("clicked", self._on_allow)
        actions.append(self.allow_btn)

        self.cancel_btn = Gtk.Button(label="Cancel")
        self.cancel_btn.set_sensitive(False)
        self.cancel_btn.connect("clicked", self._on_cancel)
        actions.append(self.cancel_btn)

        esc = Gtk.Button(label="Close")
        esc.connect("clicked", lambda *_: self.hide())
        actions.append(esc)
        box.append(actions)

        hint = Gtk.Label(
            label="Enter Send · Ctrl+Enter Send & Hide · Esc Close · Super+Space activate"
        )
        hint.add_css_class("dim-label")
        hint.set_xalign(0.0)
        box.append(hint)

        toolbar.set_content(box)
        self.set_content(toolbar)

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self.add_controller(key)

    def _on_key(self, _controller, keyval, _keycode, state) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.hide()
            return True
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and (
            state & Gdk.ModifierType.CONTROL_MASK
        ):
            self._submit(hide_after=True)
            return True
        return False

    def _on_map(self, *_args) -> None:
        _apply_launcher_theme()
        self.entry.grab_focus()
        self._refresh_projects()
        self._poll_status()

    def _selected_project(self) -> str | None:
        idx = self.project_dropdown.get_selected()
        if not self._projects or idx is None:
            return None
        # index 0 = (none)
        if idx == 0:
            return None
        if idx - 1 < len(self._projects):
            return self._projects[idx - 1]["path"]
        return None

    def _refresh_projects(self) -> None:
        def work() -> None:
            try:
                with httpx.Client(timeout=3.0) as client:
                    data = client.get(f"{GATEWAY}/api/projects").json()
                projects = data.get("projects", [])
            except Exception:  # noqa: BLE001
                projects = []
            GLib.idle_add(self._apply_projects, projects)

        threading.Thread(target=work, daemon=True).start()

    def _apply_projects(self, projects: list[dict[str, str]]) -> None:
        self._projects = projects
        names = ["(none)"] + [p["name"] for p in projects]
        model = Gtk.StringList.new(names)
        self.project_dropdown.set_model(model)
        return False

    def _poll_status(self) -> bool:
        def work() -> None:
            try:
                with httpx.Client(timeout=3.0) as client:
                    data = client.get(f"{GATEWAY}/api/status").json()
            except Exception:  # noqa: BLE001
                data = {
                    "npu": False,
                    "lmstudio": False,
                    "comfyui": False,
                    "claude": False,
                    "cursor": False,
                    "gateway": False,
                }
            GLib.idle_add(self._apply_status, data)

        threading.Thread(target=work, daemon=True).start()
        return True

    def _apply_status(self, data: dict[str, Any]) -> None:
        self._status = data
        self.local_only.set_active(bool(data.get("local_only")))

        def dot(ok: bool, *, warn: bool = False) -> str:
            if ok:
                return "●"
            if warn:
                return "◐"
            return "○"

        npu_ok = bool(data.get("npu"))
        npu_driver = bool(data.get("npu_driver"))
        self.status_label.set_text(
            "  ".join(
                [
                    f"NPU {dot(npu_ok, warn=npu_driver and not npu_ok)}",
                    f"LM Studio {dot(data.get('lmstudio', False))}",
                    f"ComfyUI {dot(data.get('comfyui', False))}",
                    f"Claude {dot(data.get('claude', False))}",
                    f"Cursor {dot(data.get('cursor', False))}",
                    f"Gateway {dot(data.get('gateway', True))}",
                ]
            )
        )
        return False

    def _set_progress(self, text: str) -> None:
        self.progress.set_text(text)

    def _on_send(self, *_args) -> None:
        self._submit(hide_after=False)

    def _on_send_hide(self, *_args) -> None:
        self._submit(hide_after=True)

    def _submit(self, *, hide_after: bool) -> None:
        message = self.entry.get_text().strip()
        if not message:
            return
        self._state = "ROUTING"
        self._set_progress("Routing…")
        self.allow_btn.set_sensitive(False)
        self.cancel_btn.set_sensitive(True)
        payload = {
            "message": message,
            "project": self._selected_project(),
            "local_only": self.local_only.get_active(),
            "confirmed": False,
        }

        def work() -> None:
            try:
                with httpx.Client(timeout=600.0) as client:
                    response = client.post(f"{GATEWAY}/api/chat", json=payload)
                    data = response.json()
            except Exception as exc:  # noqa: BLE001
                data = {"status": "failed", "error": str(exc), "task_id": ""}
            GLib.idle_add(self._on_chat_result, data, hide_after)

        threading.Thread(target=work, daemon=True).start()
        if hide_after:
            self.hide()

    def _on_chat_result(self, data: dict[str, Any], hide_after: bool) -> None:
        status = data.get("status")
        self._pending_task_id = data.get("task_id")
        if status == "waiting_confirmation":
            self._state = "WAITING_CONFIRMATION"
            self._set_progress(data.get("result") or "Confirmation required.")
            self.allow_btn.set_sensitive(True)
            self.cancel_btn.set_sensitive(True)
            self.present()
            return False

        if status == "completed":
            self._state = "SUCCESS"
            worker = data.get("worker") or "?"
            result = data.get("result") or "Done"
            self._set_progress(f"✓ Completed via {worker}\n{result}")
            self.entry.set_text("")
            self.allow_btn.set_sensitive(False)
            self.cancel_btn.set_sensitive(False)
            return False

        self._state = "FAILURE"
        err = data.get("error") or data.get("result") or json.dumps(data)
        self._set_progress(f"✗ Failed\n{err}")
        self.allow_btn.set_sensitive(False)
        self.cancel_btn.set_sensitive(False)
        if hide_after:
            self.present()
        return False

    def _on_allow(self, *_args) -> None:
        if not self._pending_task_id:
            return
        self._set_progress("Running…")
        self.allow_btn.set_sensitive(False)
        payload = {
            "message": self.entry.get_text().strip() or "confirmed",
            "project": self._selected_project(),
            "local_only": self.local_only.get_active(),
            "confirmed": True,
            "task_id": self._pending_task_id,
        }

        def work() -> None:
            try:
                with httpx.Client(timeout=600.0) as client:
                    response = client.post(f"{GATEWAY}/api/chat", json=payload)
                    data = response.json()
            except Exception as exc:  # noqa: BLE001
                data = {"status": "failed", "error": str(exc)}
            GLib.idle_add(self._on_chat_result, data, False)

        threading.Thread(target=work, daemon=True).start()

    def _on_cancel(self, *_args) -> None:
        task_id = self._pending_task_id
        if not task_id:
            self.hide()
            return

        def work() -> None:
            try:
                with httpx.Client(timeout=5.0) as client:
                    client.post(f"{GATEWAY}/api/tasks/{task_id}/cancel")
            except Exception:  # noqa: BLE001
                pass
            GLib.idle_add(self._set_progress, "Cancelled")
            GLib.idle_add(self.allow_btn.set_sensitive, False)
            GLib.idle_add(self.cancel_btn.set_sensitive, False)

        threading.Thread(target=work, daemon=True).start()

    def _on_mic(self, *_args) -> None:
        self._state = "DICTATING"
        self._set_progress("Listening (3s)…")

        def work() -> None:
            text = ""
            err = ""
            try:
                import sounddevice as sd
                import numpy as np
                from app.audio.transcription import wav_from_pcm16
                from app.config import get_config

                cfg = get_config()
                seconds = 3
                audio = sd.rec(
                    int(seconds * cfg.audio.sample_rate),
                    samplerate=cfg.audio.sample_rate,
                    channels=cfg.audio.channels,
                    dtype="int16",
                )
                sd.wait()
                wav = wav_from_pcm16(
                    audio.tobytes(),
                    sample_rate=cfg.audio.sample_rate,
                    channels=cfg.audio.channels,
                )
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        f"{GATEWAY}/api/transcribe",
                        files={"file": ("speech.wav", wav, "audio/wav")},
                    )
                    if response.status_code >= 400:
                        err = response.json().get("detail") or response.text
                    else:
                        text = response.json().get("text", "")
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def apply() -> None:
                if text:
                    current = self.entry.get_text()
                    self.entry.set_text((current + " " + text).strip())
                    self._set_progress("Transcribed.")
                else:
                    self._set_progress(f"Mic failed: {err}")
                return False

            GLib.idle_add(apply)

        threading.Thread(target=work, daemon=True).start()


class LauncherApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id="local.ai.gateway.launcher")
        self.window: LauncherWindow | None = None

    def do_activate(self) -> None:  # noqa: N802
        if self.window is None:
            self.window = LauncherWindow(self)
        self.window.present()
        self.window.entry.grab_focus()

    def do_startup(self) -> None:  # noqa: N802
        Adw.Application.do_startup(self)
        # D-Bus / desktop activation stays resident
        self.hold()


def run_launcher() -> int:
    logging.basicConfig(level=logging.INFO)
    app = LauncherApp()
    return app.run(None)
