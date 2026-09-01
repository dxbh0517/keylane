"""GTK Settings panel for Keylane."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx
from gi.repository import Gdk, GLib, Gtk  # type: ignore[attr-defined]

from ui.theme import apply_scheme_classes, effective_prefers_dark, watch_color_scheme

logger = logging.getLogger(__name__)

DAEMON = "http://127.0.0.1:9100"

_NAV = (
    ("General", "general"),
    ("Model", "model"),
    ("Web", "web"),
    ("Speech", "speech"),
    ("Skills & Tools", "skills_tools"),
    ("Security", "security"),
    ("MCP", "mcp"),
)

_THEME_LABELS = ("System", "Light", "Dark")
_THEME_VALUES = ("system", "light", "dark")


class SettingsWindow(Gtk.Window):
    def __init__(self, parent: Gtk.Window | None = None, *, independent: bool = False) -> None:
        super().__init__()
        self.set_title("Keylane Settings")
        self.set_default_size(720, 580)
        self.set_decorated(False)
        self.add_css_class("settings-window")
        self._independent = independent
        if parent and not independent:
            self.set_transient_for(parent)
            self.set_modal(True)
        else:
            self.set_modal(False)

        self._toast_cb: Any = None
        self._scheme_cb: Any = None
        self._models: list[dict[str, Any]] = []
        self._active_model_id: str | None = None
        self._loading_model = False
        self._block_save = False
        self._poll_id: int | None = None
        self._textview_provider: Gtk.CssProvider | None = None
        self._dropdown_menu_provider: Gtk.CssProvider | None = None
        self._activating_model_id: str | None = None
        self._model_rows: dict[str, Gtk.ListBoxRow] = {}
        self._model_load_progress: str = ""
        self._default_model_ids: list[str] = []

        shell = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        shell.add_css_class("settings-shell")
        self.set_child(shell)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.add_css_class("settings-header")
        title = Gtk.Label(label="Settings", xalign=0, hexpand=True)
        title.add_css_class("settings-title")
        header.append(title)
        close_btn = Gtk.Button(label="×")
        close_btn.add_css_class("settings-close")
        close_btn.set_tooltip_text("Close (Esc)")
        close_btn.connect("clicked", lambda *_: self.close())
        header.append(close_btn)
        shell.append(header)

        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        body.add_css_class("settings-body")
        body.set_vexpand(True)
        shell.append(body)

        self._nav = Gtk.ListBox()
        self._nav.add_css_class("settings-nav")
        self._nav.set_selection_mode(Gtk.SelectionMode.SINGLE)
        nav_scroll = Gtk.ScrolledWindow()
        nav_scroll.add_css_class("settings-nav-scroll")
        nav_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        nav_scroll.set_child(self._nav)
        body.append(nav_scroll)

        self._stack = Gtk.Stack()
        self._stack.add_css_class("settings-stack")
        self._stack.set_transition_type(Gtk.StackTransitionType.CROSSFADE)
        self._stack.set_transition_duration(160)
        self._stack.set_vexpand(True)
        self._stack.set_hexpand(True)
        body.append(self._stack)

        for label, page_id in _NAV:
            row = Gtk.ListBoxRow()
            row.add_css_class("settings-nav-row")
            row.set_child(Gtk.Label(label=label, xalign=0, css_classes=["settings-nav-label"]))
            row.page_id = page_id  # type: ignore[attr-defined]
            self._nav.append(row)

        self._nav.connect("row-selected", self._on_nav_selected)

        self._footer = Gtk.Label(label="", xalign=0)
        self._footer.add_css_class("settings-footer")
        shell.append(self._footer)

        self._build_general()
        self._build_model()
        self._build_web()
        self._build_speech()
        self._build_skills_tools()
        self._build_security()
        self._build_mcp()

        first = self._nav.get_row_at_index(0)
        if first:
            self._nav.select_row(first)

        watch_color_scheme(lambda _dark: self._apply_theme())
        self._apply_theme()

        self.connect("close-request", self._on_close)

    def present_centered(self) -> None:
        """Show settings; layer-shell parents need an independent toplevel."""
        self.present()
        self.set_visible(True)

        def _raise(_win: Gtk.Window, *_args: object) -> None:
            _win.set_visible(True)
            try:
                surface = _win.get_surface()
                if surface is not None:
                    surface.set_input_region(None)
            except Exception:  # noqa: BLE001
                pass

        if self.get_realized():
            _raise(self)
        else:
            self.connect("realize", _raise, once=True)

        key = Gtk.EventControllerKey.new()
        key.connect("key-released", self._on_key)
        self.add_controller(key)

    def _on_key(self, _ctrl: Gtk.EventControllerKey, keyval: int, _keycode: int, _state) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True
        return False

    def set_toast_callback(self, cb: Any) -> None:
        self._toast_cb = cb

    def set_scheme_callback(self, cb: Any) -> None:
        self._scheme_cb = cb

    def _apply_theme(self) -> None:
        apply_scheme_classes(self)
        dark = effective_prefers_dark()
        for widget in (getattr(self, "_allowlist", None),):
            if widget is None:
                continue
            widget.remove_css_class("style-light")
            widget.remove_css_class("style-dark")
            widget.add_css_class("style-dark" if dark else "style-light")
        self._apply_textview_theme()
        self._apply_default_model_menu_theme()
        popover = getattr(self, "_default_model_popover", None)
        if popover is not None:
            popover.remove_css_class("style-light")
            popover.remove_css_class("style-dark")
            popover.add_css_class("style-dark" if dark else "style-light")
        if self._scheme_cb:
            self._scheme_cb()

    def _attach_css_provider(self, provider: Gtk.CssProvider, widget: Gtk.Widget) -> None:
        widget.get_style_context().add_provider(provider, Gtk.STYLE_PROVIDER_PRIORITY_USER)

    def _apply_default_model_menu_theme(self) -> None:
        if not hasattr(self, "_default_model_menu_box"):
            return
        dark = effective_prefers_dark()
        if dark:
            css = """
popover.settings-dropdown-menu {
  background-color: #2a2a2c;
  color: #f4f4f5;
  border: 1px solid rgba(255, 255, 255, 0.12);
}
.settings-dropdown-menu-box,
.settings-dropdown-scroll,
.settings-dropdown-scroll viewport {
  background-color: #2a2a2c;
  color: #f4f4f5;
}
button.settings-dropdown-option {
  background-color: #2a2a2c;
  color: #f4f4f5;
  border: none;
  border-radius: 0;
  min-height: 34px;
  padding: 0 12px;
  box-shadow: none;
}
button.settings-dropdown-option:hover {
  background-color: rgba(255, 255, 255, 0.06);
}
button.settings-dropdown-option.selected-default {
  background-color: rgba(255, 255, 255, 0.1);
}
button.settings-dropdown-option label {
  color: #f4f4f5;
}
button.settings-dropdown-trigger {
  background-color: #2a2a2c;
  color: #f4f4f5;
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 8px;
  min-height: 36px;
  padding: 0 12px;
  box-shadow: none;
}
button.settings-dropdown-trigger:hover {
  background-color: #323234;
}
button.settings-dropdown-trigger label {
  color: #f4f4f5;
}
label.settings-dropdown-chevron {
  color: #a1a1aa;
}
"""
        else:
            css = """
popover.settings-dropdown-menu {
  background-color: #ffffff;
  color: #18181b;
  border: 1px solid rgba(0, 0, 0, 0.1);
}
.settings-dropdown-menu-box,
.settings-dropdown-scroll,
.settings-dropdown-scroll viewport {
  background-color: #ffffff;
  color: #18181b;
}
button.settings-dropdown-option {
  background-color: #ffffff;
  color: #18181b;
  border: none;
  border-radius: 0;
  min-height: 34px;
  padding: 0 12px;
  box-shadow: none;
}
button.settings-dropdown-option label {
  color: #18181b;
}
button.settings-dropdown-trigger {
  background-color: #ffffff;
  color: #18181b;
  border: 1px solid rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  min-height: 36px;
  padding: 0 12px;
  box-shadow: none;
}
button.settings-dropdown-trigger label {
  color: #18181b;
}
label.settings-dropdown-chevron {
  color: #71717a;
}
"""
        if self._dropdown_menu_provider is None:
            self._dropdown_menu_provider = Gtk.CssProvider()
            for widget in (
                self._default_model_trigger,
                self._default_model_popover,
                self._default_model_menu_box,
            ):
                self._attach_css_provider(self._dropdown_menu_provider, widget)
            scroll = self._default_model_popover.get_first_child()
            if scroll is not None:
                self._attach_css_provider(self._dropdown_menu_provider, scroll)
        self._dropdown_menu_provider.load_from_string(css)

    def _apply_textview_theme(self) -> None:
        if not hasattr(self, "_allowlist"):
            return
        if dark := effective_prefers_dark():
            css = (
                "textview.settings-textview, textview.settings-textview text {"
                "background-color: #2a2a2c; color: #e4e4e7;"
                "}"
            )
        else:
            css = (
                "textview.settings-textview, textview.settings-textview text {"
                "background-color: #ffffff; color: #27272a;"
                "}"
            )
        if self._textview_provider is None:
            self._textview_provider = Gtk.CssProvider()
            self._allowlist.get_style_context().add_provider(
                self._textview_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_USER,
            )
        self._textview_provider.load_from_string(css)

    def _toast(self, message: str) -> None:
        self._footer.set_text(message[:72])
        if self._toast_cb:
            self._toast_cb(message)

    def _patch(self, section: str, values: dict[str, Any]) -> None:
        if self._block_save:
            return
        try:
            httpx.patch(
                f"{DAEMON}/settings",
                json={"section": section, "values": values},
                timeout=10,
            ).raise_for_status()
            self._toast("Saved")
            if section == "ui":
                self._apply_theme()
        except Exception as exc:  # noqa: BLE001
            self._toast(f"Error: {exc}")

    def _on_close(self, *_args) -> bool:
        if self._poll_id:
            GLib.source_remove(self._poll_id)
            self._poll_id = None
        return False

    def _on_nav_selected(self, _nav: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if row is None:
            return
        page_id = getattr(row, "page_id", None)
        if page_id:
            self._stack.set_visible_child_name(page_id)
            if page_id == "model":
                self._load_models()
            elif page_id == "skills_tools":
                self._load_skills_tools()
            elif page_id == "mcp":
                self._load_mcp_servers()

    def _page(self, page_id: str) -> Gtk.Box:
        scroll = Gtk.ScrolledWindow()
        scroll.add_css_class("settings-content-scroll")
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_vexpand(True)
        scroll.set_hexpand(True)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        outer.add_css_class("settings-page")
        outer.set_valign(Gtk.Align.START)
        scroll.set_child(outer)
        self._stack.add_named(scroll, page_id)
        return outer

    def _section(self, parent: Gtk.Box, title: str, description: str = "") -> Gtk.Box:
        block = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        block.add_css_class("settings-section")

        heading = Gtk.Label(label=title, xalign=0)
        heading.add_css_class("settings-section-title")
        block.append(heading)

        if description:
            desc = Gtk.Label(label=description, xalign=0, wrap=True)
            desc.add_css_class("settings-section-desc")
            block.append(desc)

        parent.append(block)
        return block

    def _field(self, section: Gtk.Box, label: str, widget: Gtk.Widget, hint: str = "") -> None:
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        row.add_css_class("settings-field")

        lbl = Gtk.Label(label=label, xalign=0)
        lbl.add_css_class("settings-field-label")
        row.append(lbl)

        widget.add_css_class("settings-control")
        row.append(widget)

        if hint:
            hint_lbl = Gtk.Label(label=hint, xalign=0, wrap=True)
            hint_lbl.add_css_class("settings-field-hint")
            row.append(hint_lbl)

        section.append(row)

    def _secondary_btn(self, label: str) -> Gtk.Button:
        btn = Gtk.Button(label=label)
        btn.add_css_class("settings-btn-secondary")
        return btn

    def _primary_btn(self, label: str) -> Gtk.Button:
        btn = Gtk.Button(label=label)
        btn.add_css_class("settings-btn-primary")
        return btn

    def _badge(self, text: str, kind: str = "muted") -> Gtk.Label:
        badge = Gtk.Label(label=text)
        badge.add_css_class("settings-badge")
        badge.add_css_class(f"settings-badge-{kind}")
        return badge

    def _build_general(self) -> None:
        page = self._page("general")
        section = self._section(
            page,
            "Assistant",
            "How Keylane refers to itself and to you, and how hard it works per request.",
        )

        self._name_entry = Gtk.Entry()
        self._name_entry.set_placeholder_text("Keylane")
        self._name_entry.add_css_class("settings-entry")
        self._name_entry.connect(
            "changed",
            lambda *_: self._patch("assistant", {"name": self._name_entry.get_text()}),
        )
        self._field(section, "Assistant name", self._name_entry)

        self._user_name_entry = Gtk.Entry()
        self._user_name_entry.set_placeholder_text("Your first name")
        self._user_name_entry.add_css_class("settings-entry")
        self._user_name_entry.connect(
            "changed",
            lambda *_: self._patch("assistant", {"user_name": self._user_name_entry.get_text()}),
        )
        self._field(
            section,
            "Your name",
            self._user_name_entry,
            "Given to the model so it can address you naturally. Leave blank to stay anonymous.",
        )

        adj = Gtk.Adjustment(lower=1, upper=30, step_increment=1, page_increment=1, value=12)
        self._budget_spin = Gtk.SpinButton.new(adj, 1, 0)
        self._budget_spin.add_css_class("settings-spin")
        self._budget_spin.connect(
            "value-changed",
            lambda *_: self._patch("assistant", {"iteration_budget": int(self._budget_spin.get_value())}),
        )
        self._field(
            section,
            "Iteration budget",
            self._budget_spin,
            "Maximum tool/reasoning steps per request.",
        )

        appearance = self._section(
            page,
            "Appearance",
            "Override the system light/dark preference for Keylane.",
        )
        theme_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        theme_box.add_css_class("settings-segmented")
        self._theme_buttons: list[Gtk.ToggleButton] = []
        group: Gtk.ToggleButton | None = None
        for label in _THEME_LABELS:
            btn = Gtk.ToggleButton(label=label)
            btn.add_css_class("settings-segment")
            if group is None:
                group = btn
            else:
                btn.set_group(group)
            btn.connect("toggled", self._on_theme_toggled)
            self._theme_buttons.append(btn)
            theme_box.append(btn)
        self._field(appearance, "Theme", theme_box, "System follows your desktop setting.")

    def _on_theme_toggled(self, button: Gtk.ToggleButton) -> None:
        if not button.get_active() or self._block_save:
            return
        for i, btn in enumerate(self._theme_buttons):
            if btn is button and i < len(_THEME_VALUES):
                self._patch("ui", {"theme": _THEME_VALUES[i]})
                break

    def _model_display_name(self, model_id: str) -> str:
        for model in self._models:
            if model.get("id") == model_id:
                return str(model.get("name") or model_id)
        return model_id

    def _refresh_model_ui(self) -> None:
        self._sync_model_list(self._models)

    def _sync_default_model_menu(self, selected_id: str | None = None) -> None:
        labels: list[str] = []
        ids: list[str] = []
        for model in self._models:
            name = str(model.get("name") or model.get("id", "?"))
            if not model.get("downloaded"):
                name = f"{name} (not downloaded)"
            labels.append(name)
            ids.append(str(model["id"]))
        self._default_model_ids = ids

        child = self._default_model_menu_box.get_first_child()
        while child:
            nxt = child.get_next_sibling()
            self._default_model_menu_box.remove(child)
            child = nxt

        if not ids:
            self._default_model_label.set_text("No models")
            return

        pick = selected_id if selected_id in ids else None
        if pick is None and self._active_model_id in ids:
            pick = self._active_model_id
        if pick is None:
            pick = ids[0]

        for mid, label in zip(ids, labels):
            btn = Gtk.Button()
            btn.add_css_class("settings-dropdown-option")
            btn.set_hexpand(True)
            btn.set_halign(Gtk.Align.FILL)
            btn.set_child(Gtk.Label(label=label, xalign=0, hexpand=True))
            if mid == pick:
                btn.add_css_class("selected-default")
            btn.connect("clicked", self._on_default_model_option_clicked, mid)
            self._default_model_menu_box.append(btn)

        self._default_model_label.set_text(self._default_model_menu_label(pick))

    def _default_model_menu_label(self, model_id: str) -> str:
        display = self._model_display_name(model_id)
        for model in self._models:
            if model.get("id") == model_id and not model.get("downloaded"):
                return f"{display} (not downloaded)"
        return display

    def _pick_default_model(self, model_id: str) -> None:
        self._default_model_popover.popdown()
        idx = 0
        child = self._default_model_menu_box.get_first_child()
        while child:
            child.remove_css_class("selected-default")
            if idx < len(self._default_model_ids) and self._default_model_ids[idx] == model_id:
                child.add_css_class("selected-default")
            idx += 1
            child = child.get_next_sibling()
        self._default_model_label.set_text(self._default_model_menu_label(model_id))
        self._patch("models", {"default_model_id": model_id})
        self._toast(f"Default model set to {self._model_display_name(model_id)}")

    def _on_default_model_option_clicked(self, _btn: Gtk.Button, model_id: str) -> None:
        if self._block_save:
            return
        self._pick_default_model(model_id)

    def _set_activation_progress(self, message: str) -> None:
        text = (message or "").strip()
        if not text:
            return
        self._model_load_progress = text
        self._footer.set_text(text[:72])
        if self._activating_model_id:
            self._refresh_model_ui()
        return False

    def _build_model(self) -> None:
        page = self._page("model")
        startup = self._section(
            page,
            "Startup",
            "Which model the daemon loads automatically when it starts.",
        )
        self._default_model_trigger = Gtk.Button()
        self._default_model_trigger.add_css_class("settings-dropdown-trigger")
        self._default_model_trigger.add_css_class("settings-control")
        self._default_model_trigger.set_halign(Gtk.Align.FILL)
        self._default_model_trigger.set_hexpand(True)
        trigger_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._default_model_label = Gtk.Label(label="—", xalign=0, hexpand=True)
        chevron = Gtk.Label(label="▾", xalign=1)
        chevron.add_css_class("settings-dropdown-chevron")
        trigger_box.append(self._default_model_label)
        trigger_box.append(chevron)
        self._default_model_trigger.set_child(trigger_box)

        self._default_model_popover = Gtk.Popover()
        self._default_model_popover.add_css_class("settings-dropdown-menu")
        self._default_model_popover.set_parent(self._default_model_trigger)
        self._default_model_popover.connect("notify::visible", self._on_default_model_popover_visible)
        self._default_model_trigger.connect("clicked", lambda *_: self._default_model_popover.popup())
        menu_scroll = Gtk.ScrolledWindow()
        menu_scroll.add_css_class("settings-dropdown-scroll")
        menu_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        menu_scroll.set_max_content_height(280)
        self._default_model_menu_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self._default_model_menu_box.add_css_class("settings-dropdown-menu-box")
        menu_scroll.set_child(self._default_model_menu_box)
        self._default_model_popover.set_child(menu_scroll)

        self._field(
            startup,
            "Default model",
            self._default_model_trigger,
            "Takes effect on the next daemon restart. Undownloaded models are skipped at startup.",
        )
        self._dropdown_menu_provider = None
        self._apply_default_model_menu_theme()

        section = self._section(
            page,
            "Models",
            "Downloads continue in the background — you can close Settings while they run.",
        )

        self._model_list = Gtk.ListBox()
        self._model_list.add_css_class("settings-model-list")
        self._model_list.set_selection_mode(Gtk.SelectionMode.NONE)
        self._model_list.set_can_focus(False)
        section.append(self._model_list)

    def _on_default_model_popover_visible(self, popover: Gtk.Popover, _pspec: object) -> None:
        if popover.get_visible():
            self._apply_default_model_menu_theme()

    def _model_row_box(self, model: dict[str, Any]) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.add_css_class("settings-model-row-inner")

        mid = model["id"]
        activating = mid == getattr(self, "_activating_model_id", None)

        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        name = Gtk.Label(label=model.get("name", model.get("id", "?")), xalign=0, hexpand=True)
        name.add_css_class("settings-model-name")
        top.append(name)

        if model.get("active"):
            top.append(self._badge("Active", "active"))
        elif activating:
            top.append(self._badge("Activating", "busy"))
        elif model.get("downloaded"):
            top.append(self._badge("Downloaded", "ok"))
        elif model.get("downloading"):
            top.append(self._badge("Downloading", "busy"))
        else:
            top.append(self._badge("Not downloaded", "muted"))

        box.append(top)

        meta = Gtk.Label(
            label=f"{model.get('params_b', '?')}B · {model.get('hf_repo', '')}",
            xalign=0,
            wrap=True,
        )
        meta.add_css_class("settings-field-hint")
        box.append(meta)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        actions.add_css_class("settings-model-actions")

        if model.get("downloading"):
            dl_file = model.get("download_file") or ""
            if dl_file:
                file_lbl = Gtk.Label(label=dl_file, xalign=0, ellipsize=3)
                file_lbl.add_css_class("settings-download-file")
                box.append(file_lbl)

            dl_btn = self._secondary_btn("Download")
            dl_btn.set_sensitive(False)
            actions.append(dl_btn)

            bar = Gtk.ProgressBar()
            bar.add_css_class("settings-progress")
            bar.add_css_class("settings-progress-inline")
            bar.set_hexpand(True)
            pct = model.get("download_percent")
            if isinstance(pct, (int, float)) and pct >= 0:
                bar.set_fraction(min(1.0, float(pct) / 100.0))
            else:
                bar.pulse()
            actions.append(bar)

            pct_text = f"{int(pct)}%" if isinstance(pct, (int, float)) and pct >= 0 else "…"
            pct_label = Gtk.Label(label=pct_text)
            pct_label.add_css_class("settings-pct")
            pct_label.set_width_chars(5)
            pct_label.set_xalign(1)
            actions.append(pct_label)

        elif not model.get("downloaded"):
            dl_btn = self._secondary_btn("Download")
            dl_btn.connect("clicked", lambda *_b, m=mid: self._download_model(m))
            actions.append(dl_btn)

        if model.get("downloaded") and not model.get("active"):
            if activating:
                act_btn = self._primary_btn("Activate")
                act_btn.set_sensitive(False)
                actions.append(act_btn)

                bar = Gtk.ProgressBar()
                bar.add_css_class("settings-progress")
                bar.add_css_class("settings-progress-inline")
                bar.set_hexpand(True)
                bar.pulse()
                actions.append(bar)

                progress = self._model_load_progress if activating else ""
                pct_label = Gtk.Label(label=(progress or "Starting…")[:48])
                pct_label.add_css_class("settings-pct")
                pct_label.set_xalign(0)
                pct_label.set_hexpand(True)
                pct_label.set_ellipsize(3)
                pct_label.set_wrap(True)
                actions.append(pct_label)
            else:
                switch_btn = self._primary_btn("Activate")
                switch_btn.connect("clicked", lambda *_b, m=mid: self._activate_model(m))
                actions.append(switch_btn)

        if actions.get_first_child():
            box.append(actions)

        return box

    def _build_model_row(self, model: dict[str, Any]) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row.add_css_class("settings-model-row")
        row.set_child(self._model_row_box(model))
        return row

    def _sync_model_list(self, models: list[dict[str, Any]]) -> None:
        seen: set[str] = set()
        for model in models:
            mid = model["id"]
            seen.add(mid)
            if mid in self._model_rows:
                self._model_rows[mid].set_child(self._model_row_box(model))
            else:
                row = self._build_model_row(model)
                self._model_rows[mid] = row
                self._model_list.append(row)
        for mid in list(self._model_rows):
            if mid not in seen:
                self._model_list.remove(self._model_rows[mid])
                del self._model_rows[mid]

    def _download_model(self, model_id: str) -> None:
        try:
            httpx.post(f"{DAEMON}/models/download", json={"model_id": model_id}, timeout=15).raise_for_status()
            self._toast("Download started")
            self._load_models()
            self._ensure_poll()
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc))

    def _activate_model(self, model_id: str) -> None:
        if self._loading_model:
            busy = self._model_display_name(self._activating_model_id or "")
            self._toast(f"Already loading {busy}…")
            return
        self._loading_model = True
        self._activating_model_id = model_id
        self._model_load_progress = "Starting…"
        name = self._model_display_name(model_id)
        self._toast(f"Activating {name}…")
        self._footer.set_text("Starting…")
        self._refresh_model_ui()
        self._ensure_poll()

        def _work() -> None:
            started = time.time()
            saw_loading = False
            try:
                resp = httpx.post(
                    f"{DAEMON}/models/select",
                    json={"model_id": model_id},
                    timeout=15,
                ).raise_for_status().json()
                progress = str(resp.get("progress") or "Load queued…")
                GLib.idle_add(self._set_activation_progress, progress)
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self._finish_model_load, f"Error: {exc}")
                return

            deadline = time.time() + 900
            while time.time() < deadline:
                try:
                    health = httpx.get(f"{DAEMON}/health", timeout=10).json()
                except Exception as exc:  # noqa: BLE001
                    GLib.idle_add(self._finish_model_load, f"Error: {exc}")
                    return
                npu = health.get("npu", {})
                state = npu.get("state", "?")
                progress = str(npu.get("progress") or "")
                if progress:
                    GLib.idle_add(self._set_activation_progress, progress)
                if npu.get("loading"):
                    saw_loading = True
                if npu.get("ready") and npu.get("model_id") == model_id:
                    GLib.idle_add(self._finish_model_load, f"Activated {name}")
                    return
                if state == "error":
                    GLib.idle_add(self._finish_model_load, npu.get("error") or "load failed")
                    return
                if saw_loading and not npu.get("loading") and state in {"idle", "error"}:
                    GLib.idle_add(self._finish_model_load, npu.get("error") or "load stopped")
                    return
                if not saw_loading and time.time() - started > 15:
                    if npu.get("ready") and npu.get("model_id") == model_id:
                        GLib.idle_add(self._finish_model_load, f"Activated {name}")
                        return
                    if not npu.get("loading"):
                        GLib.idle_add(
                            self._finish_model_load,
                            "Load did not start — is another model still loading?",
                        )
                        return
                time.sleep(1)
            GLib.idle_add(self._finish_model_load, "Timed out waiting for model")

        threading.Thread(target=_work, daemon=True).start()

    def _ensure_poll(self) -> None:
        if self._poll_id:
            return

        def _tick() -> bool:
            if not self.get_visible():
                return True
            self._load_models(quiet=True)
            any_busy = any(m.get("downloading") for m in self._models) or self._loading_model
            if not any_busy and not self._loading_model:
                return False
            return True

        self._poll_id = GLib.timeout_add_seconds(1, _tick)

    def _finish_model_load(self, message: str) -> None:
        self._loading_model = False
        self._activating_model_id = None
        self._model_load_progress = ""
        self._toast(message[:60])
        self._load_models()
        return False

    def _load_models(self, quiet: bool = False) -> None:
        try:
            data = httpx.get(f"{DAEMON}/models", timeout=5).json()
            health = httpx.get(f"{DAEMON}/health", timeout=5).json()
        except Exception as exc:  # noqa: BLE001
            if not quiet:
                self._toast(str(exc))
            elif self._activating_model_id:
                self._footer.set_text(f"Cannot reach daemon: {exc}"[:72])
            if self._activating_model_id or self._loading_model:
                self._refresh_model_ui()
            return

        self._models = data.get("models", [])
        self._active_model_id = data.get("active")
        self._sync_model_list(self._models)
        self._sync_default_model_menu(data.get("default"))

        npu = health.get("npu", {})
        progress = str(npu.get("progress") or "")
        if progress and (self._activating_model_id or npu.get("loading")):
            self._model_load_progress = progress
            self._footer.set_text(progress[:72])
        if npu.get("loading"):
            self._loading_model = True
            if not self._activating_model_id:
                self._activating_model_id = npu.get("model_id")
        elif not quiet and not self._activating_model_id:
            self._loading_model = False
            self._model_load_progress = ""

        if any(m.get("downloading") for m in self._models) or self._loading_model:
            self._ensure_poll()
        elif self._activating_model_id:
            self._refresh_model_ui()

    def _build_web(self) -> None:
        page = self._page("web")
        section = self._section(
            page,
            "Web search",
            "Configure search backend and optional Playwright fetch.",
        )

        backend_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        backend_box.add_css_class("settings-segmented")
        self._backend_buttons: list[Gtk.ToggleButton] = []
        group = None
        for label in ("searxng", "ddgs"):
            btn = Gtk.ToggleButton(label=label)
            btn.add_css_class("settings-segment")
            if group is None:
                group = btn
            else:
                btn.set_group(group)
            btn.connect("toggled", self._on_backend_toggled)
            self._backend_buttons.append(btn)
            backend_box.append(btn)
        self._field(section, "Search backend", backend_box)

        self._searx_entry = Gtk.Entry()
        self._searx_entry.add_css_class("settings-entry")
        self._searx_entry.connect(
            "changed",
            lambda *_: self._patch("research", {"searxng_url": self._searx_entry.get_text()}),
        )
        self._field(section, "SearXNG URL", self._searx_entry, "Default: http://127.0.0.1:8080")

        toggles = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        toggles.add_css_class("settings-toggles")

        self._playwright_check = Gtk.CheckButton(label="Enable Playwright fetch")
        self._playwright_check.add_css_class("settings-check")
        self._playwright_check.connect(
            "toggled",
            lambda *_: self._patch("research", {"playwright_enabled": self._playwright_check.get_active()}),
        )
        toggles.append(self._playwright_check)

        self._fallback_check = Gtk.CheckButton(label="Keyless fallback (DDGS)")
        self._fallback_check.add_css_class("settings-check")
        self._fallback_check.connect(
            "toggled",
            lambda *_: self._patch("research", {"keyless_fallback": self._fallback_check.get_active()}),
        )
        toggles.append(self._fallback_check)

        section.append(toggles)

        test_btn = self._secondary_btn("Test SearXNG")
        test_btn.connect("clicked", self._test_searx)
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        actions.add_css_class("settings-actions")
        actions.append(test_btn)
        section.append(actions)

    def _on_backend_toggled(self, button: Gtk.ToggleButton) -> None:
        if not button.get_active() or self._block_save:
            return
        for i, btn in enumerate(self._backend_buttons):
            if btn is button:
                self._patch("research", {"search_backend": ["searxng", "ddgs"][i]})
                break

    def _build_speech(self) -> None:
        page = self._page("speech")
        section = self._section(page, "Speech", "Text-to-speech and read-aloud options.")

        toggles = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        toggles.add_css_class("settings-toggles")

        self._tts_notify = Gtk.CheckButton(label="TTS on notifications")
        self._tts_notify.add_css_class("settings-check")
        self._tts_notify.connect(
            "toggled",
            lambda *_: self._patch("notify", {"tts_on_notify": self._tts_notify.get_active()}),
        )
        toggles.append(self._tts_notify)

        self._read_aloud = Gtk.CheckButton(label="Read answers aloud")
        self._read_aloud.add_css_class("settings-check")
        self._read_aloud.connect(
            "toggled",
            lambda *_: self._patch("speech", {"read_aloud": self._read_aloud.get_active()}),
        )
        toggles.append(self._read_aloud)

        section.append(toggles)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        actions.add_css_class("settings-actions")
        tts_btn = self._secondary_btn("Test TTS")
        tts_btn.connect("clicked", self._test_tts)
        actions.append(tts_btn)
        notify_btn = self._secondary_btn("Test notification")
        notify_btn.connect("clicked", self._test_notify)
        actions.append(notify_btn)
        section.append(actions)

    def _build_skills_tools(self) -> None:
        page = self._page("skills_tools")
        skills_section = self._section(
            page,
            "Skills",
            "Reusable agent instructions stored in data/skills/.",
        )
        self._skills_list = Gtk.ListBox()
        self._skills_list.add_css_class("settings-item-list")
        self._skills_list.set_selection_mode(Gtk.SelectionMode.NONE)
        skills_section.append(self._skills_list)

        tools_section = self._section(
            page,
            "Tools",
            "Built-in and MCP tools available to the agent.",
        )
        self._tools_list = Gtk.ListBox()
        self._tools_list.add_css_class("settings-item-list")
        self._tools_list.set_selection_mode(Gtk.SelectionMode.NONE)
        tools_section.append(self._tools_list)

    def _load_skills_tools(self) -> None:
        try:
            skills = httpx.get(f"{DAEMON}/skills", timeout=5).json().get("skills", [])
            tools = httpx.get(f"{DAEMON}/tools", timeout=5).json().get("tools", [])
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc))
            return

        for lst, items, builder in (
            (self._skills_list, skills, self._build_skill_row),
            (self._tools_list, tools, self._build_tool_row),
        ):
            child = lst.get_first_child()
            while child:
                nxt = child.get_next_sibling()
                lst.remove(child)
                child = nxt
            if not items:
                empty = Gtk.Label(label="(none)", xalign=0)
                empty.add_css_class("settings-field-hint")
                row = Gtk.ListBoxRow()
                row.set_child(empty)
                lst.append(row)
            else:
                for item in items:
                    lst.append(builder(item))

    def _build_skill_row(self, skill: dict[str, Any]) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row.add_css_class("settings-item-row")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        title = Gtk.Label(label=skill.get("name") or skill.get("id", "?"), xalign=0)
        title.add_css_class("settings-item-title")
        box.append(title)
        desc = skill.get("description") or "No description"
        box.append(Gtk.Label(label=desc, xalign=0, wrap=True, css_classes=["settings-field-hint"]))
        row.set_child(box)
        return row

    def _build_tool_row(self, tool: dict[str, Any]) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row.add_css_class("settings-item-row")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        title = Gtk.Label(label=tool.get("name", "?"), xalign=0, hexpand=True)
        title.add_css_class("settings-item-title")
        top.append(title)
        source = tool.get("source", "builtin")
        top.append(self._badge(source, "muted"))
        if tool.get("dangerous"):
            top.append(self._badge("Dangerous", "warn"))
        box.append(top)
        box.append(
            Gtk.Label(
                label=tool.get("description", ""),
                xalign=0,
                wrap=True,
                css_classes=["settings-field-hint"],
            )
        )
        row.set_child(box)
        return row

    def _build_security(self) -> None:
        page = self._page("security")
        section = self._section(
            page,
            "Security",
            "Shell permissions and command allowlist.",
        )

        self._allowlist = Gtk.TextView()
        self._allowlist.add_css_class("settings-textview")
        self._allowlist.set_size_request(-1, 120)
        self._allowlist.set_left_margin(12)
        self._allowlist.set_right_margin(12)
        self._allowlist.set_top_margin(10)
        self._allowlist.set_bottom_margin(10)
        scroll = Gtk.ScrolledWindow()
        scroll.add_css_class("settings-text-scroll")
        scroll.set_child(self._allowlist)
        self._field(
            section,
            "Shell allowlist",
            scroll,
            "One command per line. Only listed commands may run when shell is restricted.",
        )
        self._allowlist.get_buffer().connect("changed", lambda *_: self._save_allowlist())

        perm_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        perm_box.add_css_class("settings-segmented")
        self._shell_perm_buttons: list[Gtk.ToggleButton] = []
        group = None
        for label in ("auto", "ask", "deny"):
            btn = Gtk.ToggleButton(label=label)
            btn.add_css_class("settings-segment")
            if group is None:
                group = btn
            else:
                btn.set_group(group)
            btn.connect("toggled", self._on_shell_perm_toggled)
            self._shell_perm_buttons.append(btn)
            perm_box.append(btn)
        self._field(section, "Shell permission mode", perm_box)

    def _on_shell_perm_toggled(self, button: Gtk.ToggleButton) -> None:
        if not button.get_active() or self._block_save:
            return
        modes = ["auto", "ask", "deny"]
        for i, btn in enumerate(self._shell_perm_buttons):
            if btn is button and i < len(modes):
                self._patch("permissions", {"shell": modes[i]})
                break

    def _build_mcp(self) -> None:
        page = self._page("mcp")
        section = self._section(
            page,
            "MCP servers",
            "Add stdio MCP servers. Tools appear as mcp.<id>.<tool_name>.",
        )

        self._mcp_list = Gtk.ListBox()
        self._mcp_list.add_css_class("settings-item-list")
        self._mcp_list.set_selection_mode(Gtk.SelectionMode.NONE)
        section.append(self._mcp_list)

        form = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        form.add_css_class("settings-mcp-form")

        self._mcp_id = Gtk.Entry()
        self._mcp_id.set_placeholder_text("server-id")
        self._mcp_id.add_css_class("settings-entry")
        self._field(form, "Server ID", self._mcp_id)

        self._mcp_cmd = Gtk.Entry()
        self._mcp_cmd.set_placeholder_text("npx")
        self._mcp_cmd.add_css_class("settings-entry")
        self._field(form, "Command", self._mcp_cmd)

        self._mcp_args = Gtk.Entry()
        self._mcp_args.set_placeholder_text("-y, @modelcontextprotocol/server-filesystem, /home/user")
        self._mcp_args.add_css_class("settings-entry")
        self._field(form, "Args (comma-separated)", self._mcp_args)

        add_btn = self._primary_btn("Add server")
        add_btn.connect("clicked", self._add_mcp_server)
        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        actions.add_css_class("settings-actions")
        actions.append(add_btn)
        reload_btn = self._secondary_btn("Reload tools")
        reload_btn.connect("clicked", self._reload_mcp)
        actions.append(reload_btn)
        form.append(actions)
        section.append(form)

    def _load_mcp_servers(self) -> None:
        try:
            servers = httpx.get(f"{DAEMON}/mcp/servers", timeout=15).json().get("servers", [])
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc))
            return

        child = self._mcp_list.get_first_child()
        while child:
            nxt = child.get_next_sibling()
            self._mcp_list.remove(child)
            child = nxt

        if not servers:
            row = Gtk.ListBoxRow()
            row.set_child(Gtk.Label(label="No MCP servers configured.", xalign=0, css_classes=["settings-field-hint"]))
            self._mcp_list.append(row)
            return

        for srv in servers:
            self._mcp_list.append(self._build_mcp_row(srv))

    def _build_mcp_row(self, srv: dict[str, Any]) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row.add_css_class("settings-item-row")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)

        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        sid = srv.get("id", "mcp")
        title = Gtk.Label(label=sid, xalign=0, hexpand=True)
        title.add_css_class("settings-item-title")
        top.append(title)

        health = srv.get("health", {})
        if health.get("ok"):
            top.append(self._badge(f"{health.get('tools', 0)} tools", "ok"))
        else:
            top.append(self._badge("Offline", "warn"))

        source = srv.get("source", "config")
        top.append(self._badge(source, "muted"))

        if source == "user":
            rm = self._secondary_btn("Remove")
            rm.connect("clicked", lambda *_b, s=sid: self._remove_mcp_server(s))
            top.append(rm)

        box.append(top)
        cmd_line = f"{srv.get('command', '')} {' '.join(srv.get('args', []))}".strip()
        box.append(Gtk.Label(label=cmd_line, xalign=0, wrap=True, css_classes=["settings-field-hint"]))
        row.set_child(box)
        return row

    def _add_mcp_server(self, *_args) -> None:
        sid = self._mcp_id.get_text().strip()
        cmd = self._mcp_cmd.get_text().strip()
        args_raw = self._mcp_args.get_text().strip()
        args = [a.strip() for a in args_raw.split(",") if a.strip()] if args_raw else []
        if not sid or not cmd:
            self._toast("ID and command required")
            return

        def _work() -> None:
            try:
                httpx.post(
                    f"{DAEMON}/mcp/servers",
                    json={"id": sid, "command": cmd, "args": args, "transport": "stdio"},
                    timeout=60,
                ).raise_for_status()
                GLib.idle_add(self._toast, "MCP server added")
                GLib.idle_add(self._load_mcp_servers)
                GLib.idle_add(self._mcp_id.set_text, "")
                GLib.idle_add(self._mcp_cmd.set_text, "")
                GLib.idle_add(self._mcp_args.set_text, "")
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self._toast, str(exc))

        threading.Thread(target=_work, daemon=True).start()

    def _remove_mcp_server(self, server_id: str) -> None:
        def _work() -> None:
            try:
                httpx.delete(f"{DAEMON}/mcp/servers/{server_id}", timeout=30).raise_for_status()
                GLib.idle_add(self._toast, "Removed")
                GLib.idle_add(self._load_mcp_servers)
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self._toast, str(exc))

        threading.Thread(target=_work, daemon=True).start()

    def _reload_mcp(self, *_args) -> None:
        def _work() -> None:
            try:
                r = httpx.post(f"{DAEMON}/mcp/reload", timeout=60).json()
                GLib.idle_add(self._toast, f"Loaded {r.get('tools_loaded', 0)} MCP tools")
                GLib.idle_add(self._load_mcp_servers)
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self._toast, str(exc))

        threading.Thread(target=_work, daemon=True).start()

    def _save_allowlist(self) -> None:
        if self._block_save:
            return
        buf = self._allowlist.get_buffer()
        start, end = buf.get_bounds()
        text = buf.get_text(start, end, False)
        commands = [ln.strip() for ln in text.splitlines() if ln.strip()]
        self._patch("security", {"shell_allowlist": commands})

    def _test_searx(self, *_args) -> None:
        try:
            r = httpx.get(f"{DAEMON}/research/health", timeout=15).json()
            ok = r.get("searxng", {}).get("ok", False)
            self._toast("SearXNG OK" if ok else "SearXNG failed")
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc))

    def _test_tts(self, *_args) -> None:
        try:
            httpx.post(f"{DAEMON}/settings/test/tts", timeout=30)
            self._toast("TTS test sent")
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc))

    def _test_notify(self, *_args) -> None:
        try:
            httpx.post(f"{DAEMON}/settings/test/notification", timeout=10)
            self._toast("Notification sent")
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc))

    def load_settings(self) -> None:
        self._block_save = True
        try:
            data = httpx.get(f"{DAEMON}/settings", timeout=5).json()
        except Exception:  # noqa: BLE001
            self._block_save = False
            return

        assistant = data.get("assistant", {})
        self._name_entry.set_text(str(assistant.get("name", "Keylane")))
        self._user_name_entry.set_text(str(assistant.get("user_name", "") or ""))
        self._budget_spin.set_value(int(assistant.get("iteration_budget", 12)))

        ui = data.get("ui", {})
        theme = str(ui.get("theme", "system"))
        if theme in _THEME_VALUES:
            idx = _THEME_VALUES.index(theme)
            for i, btn in enumerate(self._theme_buttons):
                btn.set_active(i == idx)

        research = data.get("research", {})
        backend = str(research.get("search_backend", "searxng"))
        for i, name in enumerate(["searxng", "ddgs"]):
            if i < len(self._backend_buttons):
                self._backend_buttons[i].set_active(name == backend)
        self._searx_entry.set_text(str(research.get("searxng_url", "http://127.0.0.1:8080")))
        self._playwright_check.set_active(bool(research.get("playwright_enabled", False)))
        self._fallback_check.set_active(bool(research.get("keyless_fallback", True)))

        notify = data.get("notify", {})
        self._tts_notify.set_active(bool(notify.get("tts_on_notify", False)))
        speech = data.get("speech", {})
        self._read_aloud.set_active(bool(speech.get("read_aloud", False)))

        security = data.get("security", {})
        allowlist = security.get("shell_allowlist", [])
        self._allowlist.get_buffer().set_text("\n".join(allowlist))

        perms = data.get("permissions", {})
        modes = ["auto", "ask", "deny"]
        shell_mode = str(perms.get("shell", "ask"))
        if shell_mode in modes:
            idx = modes.index(shell_mode)
            for i, btn in enumerate(self._shell_perm_buttons):
                btn.set_active(i == idx)

        self._block_save = False
        self._apply_theme()
        self._load_models()
        self._load_mcp_servers()
