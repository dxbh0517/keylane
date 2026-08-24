"""The Keylane popup — a Spotlight-style overlay whose shape the theme decides.

The same widget tree renders four ways, driven entirely by the active theme's
``[popup]`` spec:

``bar``     one input row that grows downward as results arrive (the default,
            and the closest thing to macOS Spotlight).
``panel``   the bar plus status chips, a project picker and hints.
``window``  a conventional decorated window with a header and scrollback.
``orb``     a small circle parked in a screen corner that expands on activation.

Positioning: on compositors with ``gtk4-layer-shell`` (sway, Hyprland, other
wlroots shells) we take an overlay layer surface and honour ``position`` and the
offsets exactly. GNOME's Mutter has no layer-shell protocol, so there we let the
compositor centre an undecorated window and reproduce a vertical offset with
transparent padding — which is what makes the bar sit above centre like
Spotlight does.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Adw, Gdk, GLib, Gtk  # noqa: E402

from launcher.gateway import GatewayClient  # noqa: E402
from launcher.theming import (  # noqa: E402
    ActiveTheme,
    apply_theme,
    install_icon_search_path,
    load_active_theme,
    logo_path,
)

logger = logging.getLogger(__name__)

ICON_NAME = "keylane"

# A dictation the user forgets to stop should not run until the disk fills.
MAX_RECORDING_SECONDS = 120

# gtk4-layer-shell, when the compositor supports it.
try:
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk4LayerShell as LayerShell  # type: ignore

    HAVE_LAYER_SHELL = True
except (ValueError, ImportError):  # pragma: no cover - environment dependent
    LayerShell = None  # type: ignore
    HAVE_LAYER_SHELL = False


STATUS_CHIPS = (
    ("npu", "NPU"),
    ("assistant", "Assistant"),
    ("lmstudio", "LM Studio"),
    ("comfyui", "ComfyUI"),
    ("claude", "Claude"),
    ("cursor", "Cursor"),
)


def pick_input_device() -> int | None:
    import sounddevice as sd

    try:
        default = sd.default.device
        default_in = default[0] if isinstance(default, (list, tuple)) else default
        if default_in is not None and int(default_in) >= 0:
            info = sd.query_devices(default_in)
            if int(info.get("max_input_channels") or 0) > 0:
                return int(default_in)
    except Exception:  # noqa: BLE001
        pass
    try:
        for index, info in enumerate(sd.query_devices()):
            if int(info.get("max_input_channels") or 0) > 0:
                return index
    except Exception:  # noqa: BLE001
        return None
    return None


class KeylanePopup(Gtk.ApplicationWindow):
    """One window, four shapes. Rebuilt whenever the active theme changes."""

    def __init__(
        self,
        app: Adw.Application,
        client: GatewayClient,
        *,
        on_closed=None,
        on_submit=None,
    ) -> None:
        super().__init__(application=app, title="Keylane")
        self.client = client
        self._on_closed = on_closed
        self._on_submit = on_submit
        self.theme: ActiveTheme = ActiveTheme()

        self._projects: list[dict[str, str]] = []
        self._status: dict[str, Any] = {}
        self._pending_task_id: str | None = None
        self._state = "IDLE"
        self._busy = False
        self._dismiss_armed = False
        self._chip_labels: dict[str, Gtk.Label] = {}
        self._layer_shell_active = False
        self._orb_expanded = False
        self._closing = False
        self._recording = False
        self._audio_stop: threading.Event | None = None
        self._audio_thread: threading.Thread | None = None

        self.add_css_class("keylane-popup")
        self.set_icon_name(ICON_NAME)
        self.set_hide_on_close(True)

        install_icon_search_path()
        self.reload_theme()

        self.connect("close-request", self._on_close_request)
        self.connect("notify::is-active", self._on_is_active)
        self.connect("map", lambda *_: GLib.idle_add(self._update_input_region))
        self.connect(
            "notify::default-height", lambda *_: GLib.idle_add(self._update_input_region)
        )
        GLib.timeout_add_seconds(6, self._poll_status)

    # ----------------------------------------------------------------- theme

    def reload_theme(self) -> None:
        """Fetch the active theme and rebuild the window to match its shape."""
        self.theme = load_active_theme(self.client)
        apply_theme(self.theme)
        self._apply_shape()
        self._build()

    @property
    def popup(self):
        return self.theme.popup

    def _apply_shape(self) -> None:
        spec = self.popup
        self.set_decorated(spec.decorated)
        self.set_resizable(spec.mode == "window")
        if spec.decorated:
            self.add_css_class("decorated")
        else:
            self.remove_css_class("decorated")

        width = max(320, spec.width)
        if spec.mode == "orb" and not self._orb_expanded:
            width = spec.orb_size + spec.padding * 2

        height = spec.height or -1
        if spec.mode == "orb" and not self._orb_expanded:
            height = spec.orb_size + spec.padding * 2
        self.set_default_size(width, height)

        self._setup_layer_shell()

    def _setup_layer_shell(self) -> None:
        """Take an overlay layer surface when the compositor offers one."""
        if not HAVE_LAYER_SHELL or self.get_realized():
            return
        spec = self.popup
        try:
            if not LayerShell.is_supported():  # type: ignore[union-attr]
                return
            LayerShell.init_for_window(self)  # type: ignore[union-attr]
            LayerShell.set_layer(self, LayerShell.Layer.OVERLAY)  # type: ignore[union-attr]
            LayerShell.set_keyboard_mode(  # type: ignore[union-attr]
                self, LayerShell.KeyboardMode.EXCLUSIVE
            )
            LayerShell.set_namespace(self, "keylane")  # type: ignore[union-attr]

            edges = {
                "top": (True, False, False, False),
                "bottom": (False, True, False, False),
                "left": (False, False, True, False),
                "right": (False, False, False, True),
                "top-left": (True, False, True, False),
                "top-right": (True, False, False, True),
                "bottom-left": (False, True, True, False),
                "bottom-right": (False, True, False, True),
                "center": (False, False, False, False),
            }
            top, bottom, left, right = edges.get(spec.position, (False, False, False, False))
            LayerShell.set_anchor(self, LayerShell.Edge.TOP, top)  # type: ignore[union-attr]
            LayerShell.set_anchor(self, LayerShell.Edge.BOTTOM, bottom)  # type: ignore[union-attr]
            LayerShell.set_anchor(self, LayerShell.Edge.LEFT, left)  # type: ignore[union-attr]
            LayerShell.set_anchor(self, LayerShell.Edge.RIGHT, right)  # type: ignore[union-attr]

            if top:
                LayerShell.set_margin(self, LayerShell.Edge.TOP, max(spec.offset_y, 0))  # type: ignore[union-attr]
            if bottom:
                LayerShell.set_margin(  # type: ignore[union-attr]
                    self, LayerShell.Edge.BOTTOM, max(-spec.offset_y, 0)
                )
            if left:
                LayerShell.set_margin(self, LayerShell.Edge.LEFT, max(spec.offset_x, 0))  # type: ignore[union-attr]
            if right:
                LayerShell.set_margin(  # type: ignore[union-attr]
                    self, LayerShell.Edge.RIGHT, max(-spec.offset_x, 0)
                )
            self._layer_shell_active = True
            logger.info("Popup using gtk4-layer-shell overlay (%s).", spec.position)
        except Exception as exc:  # noqa: BLE001
            logger.info("layer-shell unavailable (%s); using a plain window.", exc)
            self._layer_shell_active = False

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        spec = self.popup
        shell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        shell.add_css_class("keylane-shell")
        shell.set_hexpand(True)

        if spec.mode == "orb" and not self._orb_expanded:
            self.set_child(self._build_orb())
            return

        if spec.show_logo or spec.show_title:
            header = self._build_header()
            if header is not None:
                shell.append(header)

        shell.append(self._build_prompt_row())

        if spec.show_project_picker or spec.mode in {"panel", "window"}:
            shell.append(self._build_meta_row())

        if spec.show_status_chips:
            shell.append(self._build_chip_row())

        self.progress = None  # type: ignore[assignment]
        self._result_scroller = None
        if spec.show_results:
            shell.append(self._build_result_area())

        shell.append(self._build_action_row())

        if spec.show_hints:
            hint = Gtk.Label(
                label="Enter to send · Ctrl+Enter send and hide · Esc to close"
            )
            hint.add_css_class("keylane-hint")
            hint.set_xalign(0.0)
            shell.append(hint)

        self.set_child(self._wrap_for_offset(shell))
        self._install_key_controller()
        GLib.idle_add(self._update_input_region)

    def _wrap_for_offset(self, shell: Gtk.Widget) -> Gtk.Widget:
        """Reproduce a vertical offset on compositors that centre windows.

        Mutter centres a new undecorated window and gives us no way to move it.
        Padding the opposite side with transparent space shifts the visible
        content within that centred box, which is how the bar ends up above the
        middle of the screen like Spotlight.

        The padding is real window area, so it is excluded from the input region
        in :meth:`_update_input_region` — otherwise it would be a large
        invisible patch that swallows clicks.
        """
        spec = self.popup
        self._shell = shell
        shell.set_valign(Gtk.Align.START)

        if self._layer_shell_active or spec.decorated or not spec.offset_y:
            return shell

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.set_hexpand(True)
        spacer = Gtk.Box()
        spacer.set_size_request(-1, abs(spec.offset_y) * 2)
        spacer.set_can_target(False)

        if spec.offset_y < 0:
            box.append(shell)
            box.append(spacer)
        else:
            box.append(spacer)
            box.append(shell)
        return box

    def _update_input_region(self) -> None:
        """Clip the window's input region to the part you can actually see.

        Without this the transparent offset padding still belongs to the
        window: clicks land on Keylane, the popup does not lose focus, and it
        refuses to dismiss.
        """
        shell = getattr(self, "_shell", None)
        surface = self.get_surface()
        if shell is None or surface is None:
            return
        try:
            import cairo

            bounds = shell.compute_bounds(self)
            if bounds is None:
                return
            ok, rect = bounds
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
            logger.debug("Could not clip the input region: %s", exc)

    def _build_orb(self) -> Gtk.Widget:
        button = Gtk.Button()
        button.add_css_class("keylane-orb")
        button.set_has_frame(False)
        button.set_tooltip_text("Open Keylane")
        image = Gtk.Image.new_from_icon_name(ICON_NAME)
        image.set_pixel_size(max(24, self.popup.orb_size // 2))
        button.set_child(image)
        button.connect("clicked", self._on_orb_clicked)
        wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        wrapper.add_css_class("keylane-shell")
        wrapper.set_halign(Gtk.Align.CENTER)
        wrapper.set_valign(Gtk.Align.CENTER)
        wrapper.append(button)
        return wrapper

    def _on_orb_clicked(self, *_args) -> None:
        self._orb_expanded = True
        self._apply_shape()
        self._build()
        self.entry.grab_focus()

    def _mark(self, pixel_size: int) -> Gtk.Widget:
        """The Keylane mark at an exact size.

        Gtk.Picture grows to fill whatever space it is given — set_size_request
        is only a *minimum* — which is how the bar ended up 230px tall. Gtk.Image
        honours pixel_size exactly, and a -symbolic icon name lets GTK tint the
        glyph with the theme accent instead of pasting a dark tile on a dark bar.
        """
        image = Gtk.Image.new_from_icon_name("keylane-symbolic")
        display = Gdk.Display.get_default()
        if display is not None:
            theme = Gtk.IconTheme.get_for_display(display)
            if not theme.has_icon("keylane-symbolic"):
                image = Gtk.Image.new_from_icon_name(ICON_NAME)
        image.set_pixel_size(pixel_size)
        image.add_css_class("keylane-mark")
        image.set_valign(Gtk.Align.CENTER)
        image.set_halign(Gtk.Align.CENTER)
        return image

    def _build_header(self) -> Gtk.Widget | None:
        spec = self.popup
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.set_valign(Gtk.Align.CENTER)

        if spec.show_logo:
            box.append(self._mark(26))

        if spec.show_title:
            column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            column.set_valign(Gtk.Align.CENTER)
            name = Gtk.Label(label="Keylane")
            name.add_css_class("keylane-title")
            name.set_xalign(0.0)
            column.append(name)
            subtitle = Gtk.Label(label="Ask your computer")
            subtitle.add_css_class("keylane-subtitle")
            subtitle.set_xalign(0.0)
            column.append(subtitle)
            box.append(column)

        # In bar mode the mark sits inline with the entry, not on its own row.
        if spec.mode == "bar" and not spec.show_title:
            self._inline_logo = box
            return None
        return box

    def _build_prompt_row(self) -> Gtk.Widget:
        spec = self.popup
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        # CENTER, not the default FILL: one tall child must not stretch the rest.
        row.set_valign(Gtk.Align.CENTER)

        inline = getattr(self, "_inline_logo", None)
        if inline is not None:
            row.append(inline)
            self._inline_logo = None

        self.entry = Gtk.Entry()
        self.entry.set_placeholder_text(spec.input_placeholder)
        self.entry.set_hexpand(True)
        self.entry.set_valign(Gtk.Align.CENTER)
        self.entry.add_css_class("keylane-prompt")
        self.entry.set_has_frame(spec.mode != "bar")
        self.entry.connect("activate", self._on_send)
        row.append(self.entry)

        # A toggle: press to start dictating, press again to stop.
        self.mic_btn = Gtk.ToggleButton()
        self.mic_btn.set_icon_name("audio-input-microphone-symbolic")
        self.mic_btn.set_tooltip_text("Dictate — click to start, click again to stop")
        self.mic_btn.add_css_class("keylane-icon-btn")
        self.mic_btn.set_valign(Gtk.Align.CENTER)
        self.mic_btn.connect("toggled", self._on_mic_toggled)
        row.append(self.mic_btn)

        if spec.mode != "bar":
            self.send_btn = Gtk.Button(label="Send")
            self.send_btn.add_css_class("suggested-action")
            self.send_btn.add_css_class("keylane-send")
            self.send_btn.set_valign(Gtk.Align.CENTER)
            self.send_btn.connect("clicked", self._on_send)
            row.append(self.send_btn)
        else:
            self.send_btn = None  # type: ignore[assignment]
        return row

    def _build_meta_row(self) -> Gtk.Widget:
        spec = self.popup
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)

        self.project_combo = Gtk.ComboBoxText()
        self.project_combo.set_hexpand(True)
        self.project_combo.append_text("(no project)")
        self.project_combo.set_active(0)
        if spec.show_project_picker:
            label = Gtk.Label(label="Project")
            label.add_css_class("keylane-subtitle")
            row.append(label)
            row.append(self.project_combo)

        self.local_only = Gtk.CheckButton(label="Local only")
        row.append(self.local_only)
        return row

    def _build_chip_row(self) -> Gtk.Widget:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._chip_labels = {}
        for key, label in STATUS_CHIPS:
            chip = Gtk.Label(label=label)
            chip.add_css_class("keylane-chip")
            row.append(chip)
            self._chip_labels[key] = chip
        return row

    def _build_result_area(self) -> Gtk.Widget:
        spec = self.popup
        self.progress = Gtk.Label(label="")
        self.progress.set_xalign(0.0)
        self.progress.set_wrap(True)
        self.progress.set_selectable(True)
        self.progress.add_css_class("keylane-progress")

        scroller = Gtk.ScrolledWindow()
        scroller.add_css_class("keylane-result-view")
        scroller.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroller.set_child(self.progress)
        scroller.set_max_content_height(max(120, spec.max_height - 160))
        scroller.set_propagate_natural_height(True)
        # Empty until there is something to say — keeps the bar one row tall.
        scroller.set_visible(False)
        self._result_scroller = scroller
        return scroller

    def _build_action_row(self) -> Gtk.Widget:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.set_visible(False)

        self.allow_btn = Gtk.Button(label="Allow")
        self.allow_btn.add_css_class("suggested-action")
        self.allow_btn.connect("clicked", self._on_allow)
        row.append(self.allow_btn)

        self.cancel_btn = Gtk.Button(label="Cancel")
        self.cancel_btn.connect("clicked", self._on_cancel)
        row.append(self.cancel_btn)

        self._action_row = row
        return row

    def _install_key_controller(self) -> None:
        controller = Gtk.EventControllerKey()
        controller.connect("key-pressed", self._on_key)
        self.add_controller(controller)

    # ------------------------------------------------------------- lifecycle

    def present_popup(self) -> None:
        self.reload_theme()
        self.present()
        self.entry.grab_focus()
        self._refresh_projects()
        self._poll_status()

    def dismiss(self) -> None:
        """Close the popup outright.

        Hiding it left a live window sitting in the background holding focus
        state and stale text. Closing means the next Super+Space always opens a
        clean bar, re-reading the active theme on the way in.

        Notification happens in ``_on_close_request`` so it fires exactly once,
        whether the close came from here, the window manager, or Escape.
        """
        self._dismiss_armed = False
        if self.popup.mode == "orb" and self._orb_expanded:
            # In orb mode the collapsed dot is the resting state, not nothing.
            self._orb_expanded = False
            self._apply_shape()
            self._build()
            return
        if self._closing:
            return
        self._closing = True
        self.set_visible(False)
        self.close()

    def _on_close_request(self, *_args) -> bool:
        # Let the close proceed and tell the application once, so it drops its
        # reference and builds a fresh bar next time.
        self._closing = True
        if self._on_closed is not None:
            callback, self._on_closed = self._on_closed, None
            callback()
        return False

    def _on_is_active(self, *_args) -> None:
        if not self.popup.dismiss_on_focus_loss:
            return
        if self.is_active():
            self._dismiss_armed = False
            return
        if not self.get_visible():
            return
        # A dictation in progress must not be cancelled by a stray click.
        if self._state == "DICTATING":
            return
        self._dismiss_armed = True
        GLib.timeout_add(280, self._dismiss_if_inactive)

    def _dismiss_if_inactive(self) -> bool:
        if not self._dismiss_armed:
            return False
        if self.is_active() or not self.get_visible():
            return False
        if self._state == "DICTATING":
            return False
        combo = getattr(self, "project_combo", None)
        if combo is not None:
            try:
                if combo.get_property("popup-shown"):
                    return False
            except Exception:  # noqa: BLE001
                pass
        self.dismiss()
        return False

    def _on_key(self, _controller, keyval, _keycode, state) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.dismiss()
            return True
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and (
            state & Gdk.ModifierType.CONTROL_MASK
        ):
            self._submit(hide_after=True)
            return True
        return False

    # ---------------------------------------------------------------- status

    def _selected_project(self) -> str | None:
        combo = getattr(self, "project_combo", None)
        if combo is None or not self._projects:
            return None
        index = combo.get_active()
        if index is None or index <= 0:
            return None
        if index - 1 < len(self._projects):
            return self._projects[index - 1]["path"]
        return None

    def _refresh_projects(self) -> None:
        def work() -> None:
            projects = self.client.projects()
            GLib.idle_add(self._apply_projects, projects)

        threading.Thread(target=work, daemon=True).start()

    def _apply_projects(self, projects: list[dict[str, str]]) -> bool:
        combo = getattr(self, "project_combo", None)
        if combo is None:
            self._projects = projects
            return False
        previous = combo.get_active_text()
        self._projects = projects
        combo.remove_all()
        combo.append_text("(no project)")
        for project in projects:
            combo.append_text(project["name"])
        active = 0
        if previous and not previous.startswith("("):
            for index, project in enumerate(projects, start=1):
                if project["name"] == previous:
                    active = index
                    break
        combo.set_active(active)
        return False

    def _poll_status(self) -> bool:
        if not self._chip_labels:
            return True

        def work() -> None:
            data = self.client.status()
            GLib.idle_add(self._apply_status, data)

        threading.Thread(target=work, daemon=True).start()
        return True

    def _set_chip(self, key: str, *, on: bool = False, warn: bool = False) -> None:
        chip = self._chip_labels.get(key)
        if chip is None:
            return
        chip.remove_css_class("on")
        chip.remove_css_class("warn")
        if on:
            chip.add_css_class("on")
        elif warn:
            chip.add_css_class("warn")

    def _apply_status(self, data: dict[str, Any]) -> bool:
        self._status = data
        local_only = getattr(self, "local_only", None)
        if data.get("local_only") and local_only is not None and not local_only.get_active():
            local_only.set_active(True)

        npu_ok = bool(data.get("npu"))
        self._set_chip("npu", on=npu_ok, warn=bool(data.get("npu_driver")) and not npu_ok)
        self._set_chip("assistant", on=bool(data.get("tools_enabled")))
        for key in ("lmstudio", "comfyui", "claude", "cursor"):
            self._set_chip(key, on=bool(data.get(key)))
        return False

    # --------------------------------------------------------------- sending

    def _show_result(self, text: str) -> None:
        # A theme may set show_results = false, in which case there is no
        # result area to write into.
        progress = getattr(self, "progress", None)
        if progress is None:
            return
        progress.set_text(text)
        scroller = getattr(self, "_result_scroller", None)
        if scroller is not None:
            scroller.set_visible(bool(text.strip()))

    def _show_actions(self, visible: bool) -> None:
        row = getattr(self, "_action_row", None)
        if row is not None:
            row.set_visible(visible)

    def _on_send(self, *_args) -> None:
        self._submit(hide_after=False)

    def _submit(self, *, hide_after: bool) -> None:
        """Send the request, then get out of the way.

        The bar closes immediately and the working orb takes over, so a worker
        that runs for two minutes does not pin an input field to the screen.
        """
        message = self.entry.get_text().strip()
        if not message:
            return
        local_only = getattr(self, "local_only", None)
        payload = {
            "message": message,
            "project": self._selected_project(),
            "local_only": bool(local_only.get_active()) if local_only else False,
            "confirmed": False,
        }

        if self._on_submit is not None:
            self.entry.set_text("")
            self._on_submit(message, payload)
            self.dismiss()
            return

        # No orb available: keep the old inline behaviour.
        self._state = "ROUTING"
        self._busy = True
        self._show_result("Thinking…")
        self._show_actions(False)

        def work() -> None:
            data = self.client.chat(payload)
            GLib.idle_add(self._on_chat_result, data, hide_after)

        threading.Thread(target=work, daemon=True).start()
        if hide_after:
            self.dismiss()

    def _on_chat_result(self, data: dict[str, Any], hide_after: bool) -> bool:
        self._busy = False
        status = str(data.get("status") or "").lower()
        self._pending_task_id = data.get("task_id") or None

        if data.get("requires_confirmation") or status == "waiting_confirmation":
            self._state = "WAITING_CONFIRMATION"
            self._show_result(data.get("result") or "Confirmation required.")
            self._show_actions(True)
            self.present()
            return False

        self._show_actions(False)
        if status == "completed":
            self._state = "SUCCESS"
            worker = data.get("worker") or "assistant"
            result = data.get("result") or "Done."
            steps = data.get("assistant_steps") or []
            trail = (
                "\n\n" + " → ".join(str(s.get("tool")) for s in steps if s.get("tool"))
                if len(steps) > 1
                else ""
            )
            self._show_result(f"{result}{trail}\n\n— via {worker}")
            self.entry.set_text("")
            return False

        self._state = "FAILURE"
        error = data.get("error") or data.get("result") or "Something went wrong."
        self._show_result(f"Failed: {error}")
        if hide_after:
            self.present()
        return False

    def _on_allow(self, *_args) -> None:
        if not self._pending_task_id:
            return
        local_only = getattr(self, "local_only", None)
        self._busy = True
        self._show_result("Running…")
        self._show_actions(False)
        payload = {
            "message": self.entry.get_text().strip() or "confirmed",
            "project": self._selected_project(),
            "local_only": bool(local_only.get_active()) if local_only else False,
            "confirmed": True,
            "task_id": self._pending_task_id,
        }

        def work() -> None:
            data = self.client.chat(payload)
            GLib.idle_add(self._on_chat_result, data, False)

        threading.Thread(target=work, daemon=True).start()

    def _on_cancel(self, *_args) -> None:
        task_id = self._pending_task_id
        self._show_actions(False)
        if not task_id:
            self.dismiss()
            return

        def work() -> None:
            self.client.cancel(task_id)

            def done() -> bool:
                self._busy = False
                self._state = "IDLE"
                self._show_result("Cancelled.")
                return False

            GLib.idle_add(done)

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------- mic

    # --------------------------------------------------------------- dictation

    def _on_mic_toggled(self, button: Gtk.ToggleButton) -> None:
        if button.get_active():
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        """Begin an open-ended recording; the next click ends it."""
        if self._recording:
            return
        self._recording = True
        self._state = "DICTATING"
        self._busy = True
        self.mic_btn.add_css_class("recording")
        self._show_result("Listening… click the microphone again to stop.")

        self._audio_stop = threading.Event()
        thread = threading.Thread(target=self._record_worker, daemon=True)
        self._audio_thread = thread
        thread.start()

    def _stop_recording(self) -> None:
        if not self._recording:
            return
        # The worker owns the rest: it stops the stream, transcribes, and
        # restores the UI from the main loop.
        self._show_result("Transcribing…")
        self.mic_btn.set_sensitive(False)
        if self._audio_stop is not None:
            self._audio_stop.set()

    def _record_worker(self) -> None:
        text, err = self._capture_and_transcribe()

        def apply() -> bool:
            self._recording = False
            self._busy = False
            self._state = "IDLE"
            self.mic_btn.remove_css_class("recording")
            self.mic_btn.set_sensitive(True)
            # Reset without re-entering the handler.
            self.mic_btn.handler_block_by_func(self._on_mic_toggled)
            self.mic_btn.set_active(False)
            self.mic_btn.handler_unblock_by_func(self._on_mic_toggled)

            if text:
                current = self.entry.get_text().strip()
                self.entry.set_text(f"{current} {text}".strip())
                self._show_result("")
                self.entry.grab_focus()
                self.entry.set_position(-1)
            else:
                self._show_result(f"Microphone: {err}")
            return False

        GLib.idle_add(apply)

    def _capture_and_transcribe(self) -> tuple[str, str]:
        """Record until the stop event fires, then transcribe.

        Uses a callback stream rather than a fixed-length ``sd.rec`` so the
        recording length is the user's decision, not a constant.
        """
        stop = self._audio_stop
        if stop is None:
            return "", "no stop signal"

        try:
            import numpy as np
            import sounddevice as sd

            from app.audio.transcription import wav_from_pcm16

            device = pick_input_device()
            if device is None:
                return "", "no input device found"

            info = sd.query_devices(device)
            native_rate = int(float(info.get("default_samplerate") or 48000))
            try:
                sd.check_input_settings(
                    device=device, samplerate=16000, channels=1, dtype="int16"
                )
                capture_rate = 16000
            except Exception:  # noqa: BLE001
                capture_rate = native_rate

            chunks: list[Any] = []

            def on_audio(indata, _frames, _time, status) -> None:
                if status:
                    logger.debug("audio status: %s", status)
                chunks.append(indata.copy())

            with sd.InputStream(
                samplerate=capture_rate,
                channels=1,
                dtype="int16",
                device=device,
                callback=on_audio,
                blocksize=1024,
            ):
                # Cap it so a forgotten recording cannot run forever.
                stop.wait(timeout=MAX_RECORDING_SECONDS)

            if not chunks:
                return "", "nothing was recorded"

            pcm = np.concatenate(chunks, axis=0).reshape(-1).astype(np.float32)
            if pcm.size < capture_rate * 0.25:
                return "", "too short — hold the recording a moment longer"

            target_rate = 16000
            if capture_rate != target_rate and pcm.size > 1:
                duration = pcm.size / float(capture_rate)
                target_len = max(1, int(duration * target_rate))
                pcm = np.interp(
                    np.linspace(0, pcm.size - 1, target_len),
                    np.arange(pcm.size),
                    pcm,
                )
            pcm_i16 = np.clip(pcm, -32768, 32767).astype(np.int16)
            wav = wav_from_pcm16(pcm_i16.tobytes(), sample_rate=target_rate, channels=1)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Recording failed")
            return "", str(exc)

        return self.client.transcribe(wav)
