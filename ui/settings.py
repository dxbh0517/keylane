"""GTK Settings panel for Keylane."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx
from gi.repository import Gdk, GLib, Gtk  # type: ignore[attr-defined]

from mcpbridge.forms import (
    parse_args_field,
    parse_env_lines,
    parse_header_lines,
    server_endpoint,
    server_transport,
)
from ui import api
from ui.theme import (
    apply_scheme_classes,
    effective_prefers_dark,
    reload_spotlight_theme,
    theme_tokens,
    watch_color_scheme,
)
from ui.themes import list_themes

logger = logging.getLogger(__name__)


_NAV = (
    ("General", "general"),
    ("Model", "model"),
    ("Web", "web"),
    ("Speech", "speech"),
    ("Skills & Tools", "skills_tools"),
    ("Security", "security"),
    ("MCP", "mcp"),
)

# Focus around a freshly mapped override window bounces; don't act on that.
FOCUS_GRACE = 0.6      # seconds after showing
FOCUS_SETTLE_MS = 220  # confirm focus is really gone

# The popover's own border, which sits outside the width it is asked for.
_MENU_BORDER = 1

_THEME_LABELS = ("System", "Light", "Dark")
_THEME_VALUES = ("system", "light", "dark")

# Label shown in the picker, value sent to the daemon.
_MCP_TRANSPORTS = (
    ("Command (stdio)", "stdio"),
    ("URL (HTTP)", "http"),
)


class SettingsWindow(Gtk.Window):
    def __init__(self, parent: Gtk.Window | None = None, *, independent: bool = False) -> None:
        super().__init__()
        self.set_title("Keylane Settings")
        self.set_default_size(720, 580)
        self.set_decorated(False)
        self.add_css_class("settings-window")
        self._independent = independent
        if parent and not independent:
            # Transient so the window manager keeps this above the launcher,
            # but never modal: a modal grab swallows clicks on the launcher,
            # including the gear that opens this window, so the button that
            # got you here stops responding for as long as you are here.
            self.set_transient_for(parent)
        self.set_modal(False)

        self._toast_cb: Any = None
        self._scheme_cb: Any = None
        self._dismiss_cb: Any = None
        self._shown_at = 0.0
        self._had_focus = False
        self._key_controller: Gtk.EventControllerKey | None = None
        self._models: list[dict[str, Any]] = []
        self._active_model_id: str | None = None
        self._loading_model = False
        self._block_save = False
        self._poll_id: int | None = None
        self._textview_provider = Gtk.CssProvider()
        self._textviews: list[Gtk.TextView] = []
        self._mcp_transport = "stdio"
        self._activating_model_id: str | None = None
        self._model_rows: dict[str, Gtk.ListBoxRow] = {}
        self._model_load_progress: str = ""
        self._default_model_ids: list[str] = []
        self._adapters: list[dict[str, Any]] = []
        self._dropdown_popovers: list[Gtk.Popover] = []
        self._gpu_models: list[str] = []
        self._gpu_model_id: str = ""
        self._route_rows: dict[str, Gtk.Label] = {}
        # The runtime Settings is browsing. Models belong to one or the other,
        # so this is what the list below is filtered by.
        self._runtimes: list[dict[str, Any]] = []
        self._runtime_id: str = "openvino"
        self._runtime_buttons: dict[str, Gtk.ToggleButton] = {}
        self._device_ids: list[str] = []
        self._model_devices: dict[str, str] = {}
        self._importing = False
        # Daemon requests currently off the main loop, one per key.
        self._inflight: dict[str, bool] = {}

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
        self.connect("notify::is-active", self._on_active_changed)

    def _center_on_screen(self) -> bool:
        """Put the window in the middle of the screen.

        The parent is a small panel near the top of the screen, so a transient
        dialog inherits that position and opens high rather than centred.
        """
        from ui.placement import center_window

        display = Gdk.Display.get_default()
        if display is None:
            return False
        monitors = display.get_monitors()
        monitor = monitors.get_item(0) if monitors.get_n_items() > 0 else None
        if monitor is None:
            return False
        geom = monitor.get_geometry()
        scale = max(int(monitor.get_scale_factor()), 1)
        width, height = self.get_default_size()
        center_window(self, width, height, (geom.width, geom.height), scale)
        return False

    def present_centered(self) -> None:
        """Show settings; layer-shell parents need an independent toplevel."""
        self._shown_at = time.monotonic()
        self._had_focus = False
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
            # After realize, so there is a surface for the window manager to move.
            GLib.idle_add(self._center_on_screen)

        if self.get_realized():
            _raise(self)
        else:
            # `connect(..., once=True)` raises — PyGObject's connect takes no
            # keyword arguments — so this has to disconnect itself instead.
            def _raise_once(win: Gtk.Window, *_args: object) -> None:
                win.disconnect_by_func(_raise_once)
                _raise(win)

            self.connect("realize", _raise_once)

        if self._key_controller is None:
            # present_centered runs on every open; a controller per open would
            # stack another Escape handler each time.
            self._key_controller = Gtk.EventControllerKey.new()
            self._key_controller.connect("key-released", self._on_key)
            self.add_controller(self._key_controller)

    def _on_key(self, _ctrl: Gtk.EventControllerKey, keyval: int, _keycode: int, _state) -> bool:
        if keyval == Gdk.KEY_Escape:
            self.close()
            return True
        return False

    def set_toast_callback(self, cb: Any) -> None:
        self._toast_cb = cb

    def set_dismiss_callback(self, cb: Any) -> None:
        """Called when a click elsewhere closed the window, not the user."""
        self._dismiss_cb = cb

    def set_scheme_callback(self, cb: Any) -> None:
        self._scheme_cb = cb

    def _on_active_changed(self, *_args: object) -> None:
        """Close when focus goes elsewhere — the launcher's own rule.

        Focus bounces while a freshly mapped window settles, so this waits for
        focus to have really been held, then confirms it is really gone. An
        open dropdown is still this window's business, not a click away.
        """
        if self.get_property("is-active"):
            self._had_focus = True
            return
        if not self._had_focus or time.monotonic() - self._shown_at < FOCUS_GRACE:
            return

        # Which showing this close belongs to. Re-presenting the window sets a
        # new one, and a close scheduled for the old showing must not fire:
        # clicking the gear while the window is open loses it focus and
        # re-presents it, so without this the click closes what it just opened.
        showing = self._shown_at

        def _confirm() -> bool:
            if self._shown_at != showing:
                return False
            if time.monotonic() - self._shown_at < FOCUS_GRACE:
                return False
            if (
                not self.get_property("is-active")
                and self.get_visible()
                and not any(p.get_visible() for p in self._dropdown_popovers)
            ):
                self.close()
                if self._dismiss_cb:
                    self._dismiss_cb()
            return False

        GLib.timeout_add(FOCUS_SETTLE_MS, _confirm)

    def _apply_theme(self) -> None:
        apply_scheme_classes(self)
        dark = effective_prefers_dark()
        for widget in self._textviews:
            widget.remove_css_class("style-light")
            widget.remove_css_class("style-dark")
            widget.add_css_class("style-dark" if dark else "style-light")
        self._apply_textview_theme()
        # Every dropdown popover is its own surface, so the window's own style
        # class does not reach it — each is stamped directly.
        for popover in self._dropdown_popovers:
            popover.remove_css_class("style-light")
            popover.remove_css_class("style-dark")
            popover.add_css_class("style-dark" if dark else "style-light")
        if self._scheme_cb:
            self._scheme_cb()

    def _attach_css_provider(self, provider: Gtk.CssProvider, widget: Gtk.Widget) -> None:
        widget.get_style_context().add_provider(provider, Gtk.STYLE_PROVIDER_PRIORITY_USER)

    def _apply_textview_theme(self) -> None:
        # GTK4 paints a text view from its own theme unless a provider says
        # otherwise, so these two colours have to be handed over directly.
        tokens = theme_tokens()
        css = (
            "textview.settings-textview, textview.settings-textview text {"
            f"background-color: {tokens.get('textview-bg', '#ffffff')};"
            f" color: {tokens.get('textview-text', '#27272a')};"
            "}"
        )
        self._textview_provider.load_from_string(css)

    def _fetch_async(self, key: str, work: Any, apply: Any) -> None:
        """Run *work* off the main loop and hand the result to *apply* on it.

        Settings talks to the daemon over HTTP, and the daemon answers some of
        it slowly — /settings/health probes SearXNG and every MCP server, which
        measured over a second on a healthy machine and is bounded only by
        timeouts on an unhealthy one. Doing that on the GTK thread freezes the
        whole UI, and a frozen window is indistinguishable from a dead button:
        the click that opened Settings looks like it did nothing, so it gets
        clicked again.

        One request per key is in flight at a time, so the once-a-second poll
        during a download skips a tick rather than queueing up behind itself.
        """
        if self._inflight.get(key):
            return
        self._inflight[key] = True

        def _worker() -> None:
            try:
                result, error = work(), None
            except Exception as exc:  # noqa: BLE001
                result, error = None, exc

            def _done() -> bool:
                self._inflight[key] = False
                apply(result, error)
                return False

            GLib.idle_add(_done)

        threading.Thread(target=_worker, daemon=True, name=f"settings-{key}").start()

    def _toast(self, message: str) -> None:
        self._footer.set_text(message[:72])
        if self._toast_cb:
            self._toast_cb(message)

    def _patch(self, section: str, values: dict[str, Any]) -> None:
        if self._block_save:
            return
        try:
            api.patch(
                "/settings",
                json={"section": section, "values": values},
                timeout=10,
            ).raise_for_status()
            self._toast("Saved")
            if section == "ui":
                reload_spotlight_theme()
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
                self._load_routes()
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

    def _textview_field(
        self,
        section: Gtk.Box,
        label: str,
        hint: str = "",
        height: int = 90,
    ) -> Gtk.TextView:
        """A multi-line field, registered so the theme reaches it.

        GTK4 draws a text view with its own Adwaita colours, so each one needs
        the runtime provider; building them here means a new field is themed
        by construction rather than by remembering.
        """
        view = Gtk.TextView()
        view.add_css_class("settings-textview")
        view.set_size_request(-1, height)
        view.set_left_margin(12)
        view.set_right_margin(12)
        view.set_top_margin(10)
        view.set_bottom_margin(10)
        view.get_style_context().add_provider(
            self._textview_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_USER,
        )
        self._textviews.append(view)

        scroll = Gtk.ScrolledWindow()
        scroll.add_css_class("settings-text-scroll")
        scroll.set_child(view)
        self._field(section, label, scroll, hint)
        self._apply_theme()  # stamp it before it is ever shown
        return view

    def _entry(self, placeholder: str = "") -> Gtk.Entry:
        entry = Gtk.Entry()
        entry.add_css_class("settings-entry")
        if placeholder:
            entry.set_placeholder_text(placeholder)
        return entry

    def _make_dropdown(self, placeholder: str = "—") -> tuple[Gtk.Button, Gtk.Label, Gtk.Popover, Gtk.Box]:
        """A trigger button plus its menu popover, themed with the others.

        A popover is its own surface, so the window's style classes do not
        reach it and it needs the runtime provider. Registering here means a
        new dropdown is themed by construction rather than by remembering.
        """
        trigger = Gtk.Button()
        trigger.add_css_class("settings-dropdown-trigger")
        trigger.add_css_class("settings-control")
        trigger.set_halign(Gtk.Align.FILL)
        trigger.set_hexpand(True)

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        label = Gtk.Label(label=placeholder, xalign=0, hexpand=True)
        chevron = Gtk.Label(label="▾", xalign=1)
        chevron.add_css_class("settings-dropdown-chevron")
        box.append(label)
        box.append(chevron)
        trigger.set_child(box)

        popover = Gtk.Popover()
        popover.add_css_class("settings-dropdown-menu")
        popover.set_parent(trigger)
        trigger.connect("clicked", lambda *_: popover.popup())

        # How much wider than its request the popover renders — border, shadow
        # and whatever else the theme adds. Guessed, then measured on first use.
        chrome = [_MENU_BORDER * 2]

        def _match_trigger_width(*_args: object) -> None:
            """GTK sizes a popover to its contents, which leaves a menu
            narrower than the control it hangs off; the edges should line up."""
            width = trigger.get_width()
            if not popover.get_visible() or width <= 0:
                return
            popover.set_size_request(width - chrome[0], -1)

            def _correct() -> bool:
                extra = popover.get_width() - width
                if extra:
                    chrome[0] += extra
                    popover.set_size_request(width - chrome[0], -1)
                return False

            GLib.idle_add(_correct)

        popover.connect("notify::visible", _match_trigger_width)

        scroll = Gtk.ScrolledWindow()
        scroll.add_css_class("settings-dropdown-scroll")
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_max_content_height(280)
        menu_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        menu_box.add_css_class("settings-dropdown-menu-box")
        scroll.set_child(menu_box)
        popover.set_child(scroll)

        self._dropdown_popovers.append(popover)
        self._apply_theme()  # stamp the new popover before it is ever shown
        return trigger, label, popover, menu_box

    @staticmethod
    def _clear_box(box: Gtk.Box) -> None:
        child = box.get_first_child()
        while child:
            nxt = child.get_next_sibling()
            box.remove(child)
            child = nxt

    def _status_row(self, section: Gtk.Box, label: str, value: Gtk.Label, hint: str = "") -> None:
        """A read-only row: name on the left, current value on the right.

        `_field` stacks its widget under the label, which reads as an orphaned
        value when the widget is just text rather than a control.
        """
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        row.add_css_class("settings-field")

        line = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        lbl = Gtk.Label(label=label, xalign=0, hexpand=True)
        lbl.add_css_class("settings-field-label")
        line.append(lbl)
        value.set_xalign(1)
        line.append(value)
        row.append(line)

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

        self._auto_learn = Gtk.CheckButton(label="Let Keylane propose reusable skills")
        self._auto_learn.add_css_class("settings-check")
        self._auto_learn.connect(
            "toggled",
            lambda *_: self._patch(
                "assistant", {"auto_learn_skills": self._auto_learn.get_active()}
            ),
        )
        self._field(
            section,
            "Learn from tasks",
            self._auto_learn,
            "After a multi-step task, Keylane may offer to save what it did as a "
            "skill. It always asks before writing the file. Off by default.",
        )

        appearance = self._section(
            page,
            "Appearance",
            "Which theme Keylane wears, and whether it follows your desktop's "
            "light/dark preference.",
        )

        (
            theme_trigger,
            self._theme_label,
            self._theme_popover,
            theme_menu,
        ) = self._make_dropdown()
        self._theme_options: dict[str, Gtk.Button] = {}
        self._theme_ids: list[str] = []
        for theme in list_themes():
            option = Gtk.Button()
            option.add_css_class("settings-dropdown-option")
            option.set_hexpand(True)
            option.set_halign(Gtk.Align.FILL)
            option.set_child(Gtk.Label(label=theme.name, xalign=0, hexpand=True))
            option.connect("clicked", lambda _b, t=theme.id: self._pick_theme(t))
            theme_menu.append(option)
            self._theme_options[theme.id] = option
            self._theme_ids.append(theme.id)
        self._field(
            appearance,
            "Theme",
            theme_trigger,
            "Your own themes go in data/themes/ — see README → Themes.",
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
        self._field(appearance, "Light or dark", theme_box, "System follows your desktop setting.")

    def _pick_theme(self, theme_id: str) -> None:
        self._theme_popover.popdown()
        self._show_theme(theme_id)
        if self._block_save:
            return
        self._patch("ui", {"theme_id": theme_id})

    def _show_theme(self, theme_id: str) -> None:
        """Mark the picked theme in the menu and name it on the trigger."""
        for theme in list_themes():
            option = self._theme_options.get(theme.id)
            if option is None:
                continue
            if theme.id == theme_id:
                option.add_css_class("selected-default")
                self._theme_label.set_text(theme.name)
            else:
                option.remove_css_class("selected-default")

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
            name = f"{name} · {self._runtime_tag(str(model.get('runtime', '')))}"
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
            if model.get("id") != model_id:
                continue
            display = f"{display} · {self._runtime_tag(str(model.get('runtime', '')))}"
            if not model.get("downloaded"):
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
        self._build_runtime(page)

        startup = self._section(
            page,
            "Startup",
            "Which model the daemon loads automatically when it starts.",
        )
        (
            self._default_model_trigger,
            self._default_model_label,
            self._default_model_popover,
            self._default_model_menu_box,
        ) = self._make_dropdown()

        self._field(
            startup,
            "Default model",
            self._default_model_trigger,
            "Takes effect on the next daemon restart. Undownloaded models are skipped at startup.",
        )

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

        self._model_empty = Gtk.Label(label="", xalign=0, wrap=True)
        self._model_empty.add_css_class("settings-field-hint")
        self._model_empty.set_visible(False)
        section.append(self._model_empty)

        self._build_import(page)
        self._build_routes(page)

    # ── runtime ──────────────────────────────────────────────────────────

    def _build_runtime(self, page: Gtk.Box) -> None:
        """Which inference stack runs the local model, and on which device."""
        section = self._section(
            page,
            "Runtime",
            "A model is an export for one stack or the other, so this also "
            "decides which models you can pick below.",
        )

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        box.add_css_class("settings-segmented")
        group: Gtk.ToggleButton | None = None
        # Labelled before /runtimes answers, so the control is never empty; the
        # names are corrected from the daemon on the first load.
        for runtime_id, label in (
            ("openvino", "OpenVINO GenAI"),
            ("onnxruntime", "ONNX Runtime"),
        ):
            btn = Gtk.ToggleButton(label=label)
            btn.add_css_class("settings-segment")
            if group is None:
                group = btn
            else:
                btn.set_group(group)
            btn.connect("toggled", self._on_runtime_toggled, runtime_id)
            self._runtime_buttons[runtime_id] = btn
            box.append(btn)
        self._field(section, "Inference runtime", box)

        self._runtime_status = Gtk.Label(label="", xalign=0, wrap=True)
        self._runtime_status.add_css_class("settings-field-hint")
        section.append(self._runtime_status)

        (
            self._device_trigger,
            self._device_label,
            self._device_popover,
            self._device_menu,
        ) = self._make_dropdown("NPU")
        self._field(
            section,
            "Device",
            self._device_trigger,
            "Applies to models on this runtime. Changing it invalidates the "
            "compile cache, so the next load is slow either way.",
        )

    def _runtime_info(self, runtime_id: str) -> dict[str, Any]:
        for info in self._runtimes:
            if info.get("id") == runtime_id:
                return info
        return {}

    def _on_runtime_toggled(self, button: Gtk.ToggleButton, runtime_id: str) -> None:
        if self._block_save or not button.get_active():
            return
        self._runtime_id = runtime_id
        self._patch("models", {"runtime": runtime_id})
        self._sync_runtime_status()
        self._sync_device_menu()
        self._refresh_model_ui()
        self._sync_default_model_menu()

    def _sync_runtime_status(self) -> None:
        info = self._runtime_info(self._runtime_id)
        if not info:
            self._runtime_status.set_text("")
            return
        summary = str(info.get("summary", ""))
        if info.get("installed"):
            self._runtime_status.set_text(f"{summary} Installed: {info.get('detail', '')}")
        else:
            self._runtime_status.set_text(
                f"{summary} Not installed — {info.get('detail', '')}. "
                f"Install it with: {info.get('install_hint', '')}"
            )

    def _sync_device_menu(self) -> None:
        info = self._runtime_info(self._runtime_id)
        devices = [str(d) for d in info.get("devices", [])] or ["NPU"]
        self._device_ids = devices

        current = str(self._model_devices.get(self._runtime_id, "") or "")
        if current not in devices:
            current = str(info.get("default_device") or devices[0])
        self._device_label.set_text(current)

        # Every device the runtime knows about, usable or not. A device that is
        # present but cannot be compiled for — an NVIDIA card OpenVINO happens
        # to enumerate — is shown with its reason rather than hidden, because
        # the user can see the hardware and would otherwise wonder.
        rows = info.get("all_devices") or [
            {"id": d, "label": d, "usable": True, "reason": ""} for d in devices
        ]

        self._clear_box(self._device_menu)
        for row in rows:
            device = str(row.get("id", ""))
            usable = bool(row.get("usable", True))
            btn = Gtk.Button(label=str(row.get("label") or device))
            btn.add_css_class("settings-dropdown-option")
            if not usable:
                btn.set_sensitive(False)
                btn.set_tooltip_text(str(row.get("reason") or "unavailable"))
                self._device_menu.append(btn)
                continue
            if device == current:
                btn.add_css_class("selected-default")
            btn.connect("clicked", lambda _b, d=device: self._set_device(d))
            self._device_menu.append(btn)

    def _set_device(self, device: str) -> None:
        self._device_popover.popdown()
        self._model_devices[self._runtime_id] = device
        self._device_label.set_text(device)
        self._patch("models", {"devices": dict(self._model_devices)})
        self._toast(f"{self._runtime_label(self._runtime_id)} will use {device}")
        self._sync_device_menu()
        self._refresh_model_ui()

    def _runtime_label(self, runtime_id: str) -> str:
        info = self._runtime_info(runtime_id)
        return str(info.get("name") or runtime_id)

    def _runtime_tag(self, runtime_id: str) -> str:
        """A short name for a row that has to say which stack it belongs to."""
        return {"openvino": "OpenVINO", "onnxruntime": "ONNX"}.get(runtime_id, runtime_id)

    # ── importing from Hugging Face ──────────────────────────────────────

    def _build_import(self, page: Gtk.Box) -> None:
        section = self._section(
            page,
            "Import from Hugging Face",
            "Anything not on the curated list. Keylane reads the repo first and "
            "refuses one it could not load, so nothing is downloaded on a guess.",
        )

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._import_entry = self._entry("OpenVINO/Qwen3-8B-int4-ov")
        self._import_entry.set_hexpand(True)
        self._import_entry.connect("activate", lambda *_: self._import_model())
        row.append(self._import_entry)

        self._import_btn = self._primary_btn("Import")
        self._import_btn.connect("clicked", lambda *_: self._import_model())
        row.append(self._import_btn)

        self._field(
            section,
            "Repo id or URL",
            row,
            "Needs an OpenVINO IR export (openvino_model.xml — repos named "
            "*-int4-ov) or an ONNX Runtime GenAI export (genai_config.json). "
            "A repo with several builds is matched to this machine automatically.",
        )

    def _import_model(self) -> None:
        repo = self._import_entry.get_text().strip()
        if not repo:
            self._toast("Paste a Hugging Face repo id first")
            return
        if self._importing:
            return
        self._importing = True
        self._import_btn.set_sensitive(False)
        self._toast(f"Reading {repo}…")

        def _work() -> None:
            try:
                resp = api.post(
                    "/models/import",
                    json={"repo": repo},
                    timeout=60,
                )
                if resp.status_code >= 400:
                    detail = resp.json().get("detail", resp.text)
                    GLib.idle_add(self._finish_import, "", str(detail))
                    return
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self._finish_import, "", str(exc))
                return
            model = data.get("model", {})
            others = len([v for v in data.get("variants", []) if not v.get("chosen")])
            note = f" ({others} other build(s) in that repo)" if others else ""
            GLib.idle_add(
                self._finish_import,
                f"Imported {model.get('name', repo)} for "
                f"{self._runtime_tag(str(model.get('runtime', '')))}{note}",
                "",
                str(model.get("runtime", "")),
            )

        threading.Thread(target=_work, daemon=True).start()

    def _finish_import(self, message: str, error: str, runtime_id: str = "") -> bool:
        self._importing = False
        self._import_btn.set_sensitive(True)
        if error:
            self._toast(error[:120])
            self._footer.set_text(error[:110])
            return False
        self._import_entry.set_text("")
        self._toast(message[:80])
        # An import lands in its own runtime's list, which may not be the one
        # on screen — switch to it rather than appearing to have done nothing.
        if runtime_id and runtime_id != self._runtime_id:
            self._select_runtime(runtime_id)
        self._load_models()
        return False

    def _select_runtime(self, runtime_id: str) -> None:
        button = self._runtime_buttons.get(runtime_id)
        if button is None:
            return
        button.set_active(True)

    def _forget_model(self, model_id: str) -> None:
        try:
            api.delete(f"/models/imported/{model_id}", timeout=15).raise_for_status()
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc)[:80])
            return
        self._toast("Removed from the list — downloaded files were kept")
        self._load_models()

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

        if model.get("source") == "imported":
            top.append(self._badge("Imported", "muted"))

        box.append(top)

        params = model.get("params_b") or 0
        parts = [f"{params}B" if params else "size unknown", str(model.get("hf_repo", ""))]
        if model.get("subfolder"):
            parts.append(str(model["subfolder"]))
        if model.get("device"):
            parts.append(f"on {model['device']}")
        meta = Gtk.Label(label=" · ".join(p for p in parts if p), xalign=0, wrap=True)
        meta.add_css_class("settings-field-hint")
        box.append(meta)

        description = str(model.get("description", "") or "")
        if description:
            desc = Gtk.Label(label=description, xalign=0, wrap=True)
            desc.add_css_class("settings-field-hint")
            box.append(desc)

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

        if model.get("source") == "imported" and not model.get("downloading"):
            forget = self._secondary_btn("Forget")
            forget.set_tooltip_text("Remove from the list. Downloaded files are kept.")
            forget.connect("clicked", lambda *_b, m=mid: self._forget_model(m))
            actions.append(forget)

        if actions.get_first_child():
            box.append(actions)

        return box

    def _build_model_row(self, model: dict[str, Any]) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row.add_css_class("settings-model-row")
        row.set_child(self._model_row_box(model))
        return row

    def _sync_model_list(self, models: list[dict[str, Any]]) -> None:
        models = [m for m in models if m.get("runtime", "openvino") == self._runtime_id]
        self._sync_model_empty(len(models))
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

    def _sync_model_empty(self, shown: int) -> None:
        """Say why the list is empty rather than showing a blank box."""
        if shown:
            self._model_empty.set_visible(False)
            return
        elsewhere = len(self._models) - shown
        info = self._runtime_info(self._runtime_id)
        if not info.get("installed", True):
            text = (
                f"{self._runtime_label(self._runtime_id)} is not installed, so none "
                f"of its models can run yet. Install it with: {info.get('install_hint', '')}"
            )
        elif elsewhere:
            text = (
                f"No models for this runtime. {elsewhere} model(s) are on the other "
                "runtime — switch above, or import one below."
            )
        else:
            text = "No models yet. Import one from Hugging Face below."
        self._model_empty.set_text(text)
        self._model_empty.set_visible(True)

    def _download_model(self, model_id: str) -> None:
        try:
            api.post("/models/download", json={"model_id": model_id}, timeout=15).raise_for_status()
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
                resp = api.post(
                    "/models/select",
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
                    health = api.get("/health", timeout=10).json()
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
            data = api.get("/models", timeout=5).json()
            health = api.get("/health", timeout=5).json()
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
        self._runtimes = list(data.get("runtimes", []) or [])
        devices = data.get("devices", {})
        if isinstance(devices, dict):
            self._model_devices = {str(k): str(v) for k, v in devices.items()}
        self._sync_runtime_buttons(str(data.get("runtime", "") or ""))
        self._sync_runtime_status()
        self._sync_device_menu()
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

    def _build_routes(self, page: Gtk.Box) -> None:
        """Model routing: which model serves which kind of work."""
        section = self._section(
            page,
            "Model routing",
            "Keylane picks a model by what the work is, not by name. The spotlight "
            "answer stays on the NPU; longer work can go to a larger model.",
        )

        self._route_rows: dict[str, Gtk.Label] = {}
        for route, blurb in (
            ("interactive", "The answer you are waiting for"),
            ("background", "Subagents, scheduled work, research synthesis"),
            ("utility", "Query planning and URL selection"),
        ):
            value = Gtk.Label(label="—", xalign=1)
            value.add_css_class("settings-item-title")
            self._route_rows[route] = value
            self._status_row(section, route.capitalize(), value, blurb)

        gpu = self._section(
            page,
            "Larger model (optional)",
            "Any server speaking the OpenAI chat-completions API — LM Studio, "
            "llama.cpp, Ollama, vLLM. Once enabled it takes the background work "
            "and leaves the NPU free for the spotlight.",
        )

        self._gpu_enabled = Gtk.CheckButton(label="Use a larger model for background work")
        self._gpu_enabled.add_css_class("settings-check")
        self._gpu_enabled.connect("toggled", lambda *_: self._save_gpu_adapter())
        self._field(gpu, "Enabled", self._gpu_enabled)

        self._gpu_url = Gtk.Entry()
        self._gpu_url.set_placeholder_text("http://127.0.0.1:1234/v1")
        self._gpu_url.add_css_class("settings-entry")
        self._gpu_url.connect("changed", lambda *_: self._save_gpu_adapter())
        self._field(gpu, "Server URL", self._gpu_url, "The API base, ending in /v1.")

        (
            self._gpu_model_trigger,
            self._gpu_model_label,
            self._gpu_model_popover,
            self._gpu_model_menu,
        ) = self._make_dropdown("Connect to see models")
        # Fill the list when the menu is opened, so a server started after
        # Settings was opened still shows up without a reload.
        self._gpu_model_popover.connect("notify::visible", self._on_gpu_menu_visible)
        self._field(
            gpu,
            "Model",
            self._gpu_model_trigger,
            "Fetched from the server. Open the list to refresh it.",
        )

        self._gpu_unload = Gtk.CheckButton(label="Unload from VRAM when idle")
        self._gpu_unload.add_css_class("settings-check")
        self._gpu_unload.connect("toggled", lambda *_: self._save_gpu_adapter())
        self._field(
            gpu,
            "Free the GPU when idle",
            self._gpu_unload,
            "Keeps the model out of VRAM between tasks, at the cost of a reload "
            "on the next one. Needs a server that honours it — Ollama and "
            "LM Studio do; llama.cpp's server does not.",
        )

        test = self._secondary_btn("Test connection")
        test.connect("clicked", self._test_gpu_model)
        self._field(gpu, "", test)

    def _on_gpu_menu_visible(self, popover: Gtk.Popover, _pspec: object) -> None:
        if popover.get_visible():
            self._load_gpu_models()

    def _set_gpu_model(self, model_id: str) -> None:
        self._gpu_model_id = model_id
        self._gpu_model_label.set_text(model_id or "Connect to see models")
        self._gpu_model_popover.popdown()
        self._save_gpu_adapter()

    def _sync_gpu_menu(self) -> None:
        self._clear_box(self._gpu_model_menu)
        if not self._gpu_models:
            empty = Gtk.Label(label="No models found", xalign=0)
            empty.add_css_class("settings-field-hint")
            self._gpu_model_menu.append(empty)
            return
        for name in self._gpu_models:
            btn = Gtk.Button(label=name)
            btn.add_css_class("settings-dropdown-option")
            if name == self._gpu_model_id:
                btn.add_css_class("selected-default")
            btn.connect("clicked", lambda _b, m=name: self._set_gpu_model(m))
            self._gpu_model_menu.append(btn)

    def _fetch_gpu_models(self) -> list[str] | str:
        """Model ids the configured server reports, or an error string."""
        base = self._gpu_url.get_text().strip().rstrip("/")
        if not base:
            return "Set a server URL first"
        try:
            resp = httpx.get(f"{base}/models", timeout=8)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            return f"Cannot reach {base}: {exc}"
        rows = payload.get("data", payload if isinstance(payload, list) else [])
        return sorted(str(m.get("id", "")) for m in rows if m.get("id"))

    def _load_gpu_models(self) -> None:
        found = self._fetch_gpu_models()
        if isinstance(found, str):
            self._gpu_models = []
            self._sync_gpu_menu()
            self._toast(found[:72])
            return
        self._gpu_models = found
        # A model that vanished from the server should not stay selected.
        if self._gpu_model_id and self._gpu_model_id not in found:
            self._toast(f"{self._gpu_model_id!r} is no longer served here")
        self._sync_gpu_menu()

    def _save_gpu_adapter(self) -> None:
        if self._block_save:
            return
        # The whole adapter list is replaced, so any other adapters are carried
        # through unchanged rather than dropped by a partial write.
        adapters = [a for a in self._adapters if a.get("id") != "gpu"]
        adapters.append(
            {
                "id": "gpu",
                "kind": "openai",
                "base_url": self._gpu_url.get_text().strip(),
                "model": self._gpu_model_id,
                "enabled": self._gpu_enabled.get_active(),
                "auto_unload": self._gpu_unload.get_active(),
            }
        )
        self._adapters = adapters
        self._patch("models", {"adapters": adapters})

    def _test_gpu_model(self, *_args) -> None:
        found = self._fetch_gpu_models()
        if isinstance(found, str):
            self._toast(found[:72])
            return
        self._gpu_models = found
        self._sync_gpu_menu()
        if self._gpu_model_id and self._gpu_model_id not in found:
            self._toast(f"Reachable, but {self._gpu_model_id!r} is not served there")
        else:
            self._toast(f"OK — {len(found)} model(s) available")

    def _sync_runtime_buttons(self, runtime_id: str) -> None:
        """Name the segments from the daemon and mark the stored choice.

        Restores the guard rather than clearing it: this reflects the daemon's
        state into the widgets, so it must not decide on its caller's behalf
        that writing settings back is safe again.
        """
        was_blocked = self._block_save
        self._block_save = True
        try:
            for info in self._runtimes:
                button = self._runtime_buttons.get(str(info.get("id")))
                if button is not None:
                    button.set_label(str(info.get("name") or info.get("id")))
            if runtime_id in self._runtime_buttons:
                self._runtime_id = runtime_id
                self._runtime_buttons[runtime_id].set_active(True)
        finally:
            self._block_save = was_blocked

    def _load_routes(self) -> None:
        def _work() -> dict[str, Any]:
            return api.get("/settings/health", timeout=8).json()

        def _apply(health: dict[str, Any] | None, error: Exception | None) -> None:
            if error is not None or health is None:
                for label in self._route_rows.values():
                    label.set_text("unavailable")
                return
            routes = health.get("models", {}).get("routes", {})
            for route, label in self._route_rows.items():
                info = routes.get(route, {})
                resolved = info.get("resolved")
                preference = " → ".join(info.get("preference", []))
                label.set_text(resolved or f"none ready ({preference})")

        self._fetch_async("routes", _work, _apply)

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
            "Reusable instructions loaded only when a task needs them. Kept in "
            "data/skills/, .keylane/skills/ in a project, or the bundled skills/ "
            "folder — a <name>/SKILL.md bundle or a flat <name>.md. Type /name in "
            "the spotlight bar to apply one directly.",
        )
        self._skills_list = Gtk.ListBox()
        self._skills_list.add_css_class("settings-item-list")
        self._skills_list.set_selection_mode(Gtk.SelectionMode.NONE)
        skills_section.append(self._skills_list)

        tools_section = self._section(
            page,
            "Tools",
            "Built-in and MCP tools available to the agent. Ones marked "
            "\u201cAsks first\u201d prompt for permission before they run.",
        )
        self._tools_list = Gtk.ListBox()
        self._tools_list.add_css_class("settings-item-list")
        self._tools_list.set_selection_mode(Gtk.SelectionMode.NONE)
        tools_section.append(self._tools_list)

    def _load_skills_tools(self) -> None:
        try:
            skills = api.get("/skills", timeout=5).json().get("skills", [])
            tools = api.get("/tools", timeout=5).json().get("tools", [])
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

        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        name = skill.get("name") or skill.get("id", "?")
        title = Gtk.Label(label=name, xalign=0, hexpand=True)
        title.add_css_class("settings-item-title")
        top.append(title)
        top.append(self._badge(str(skill.get("source", "user")), "muted"))

        model_ok = skill.get("model_invocable", True)
        user_ok = skill.get("user_invocable", True)
        if not model_ok and not user_ok:
            top.append(self._badge("Off", "warn"))
        elif not model_ok:
            # Reachable only by typing /name — worth saying, since it will never
            # appear in the model's catalog.
            top.append(self._badge(f"/{name} only", "muted"))
        elif not user_ok:
            top.append(self._badge("Model only", "muted"))
        box.append(top)

        desc = skill.get("description") or "No description"
        box.append(Gtk.Label(label=desc, xalign=0, wrap=True, css_classes=["settings-field-hint"]))
        when = skill.get("when_to_use")
        if when:
            box.append(
                Gtk.Label(
                    label=f"Use when: {when}",
                    xalign=0,
                    wrap=True,
                    css_classes=["settings-field-hint"],
                )
            )
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
        if tool.get("gated"):
            # What actually matters to the user: this one goes through the
            # permission gate, whatever its dangerous flag says.
            top.append(self._badge("Asks first", "warn"))
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

        self._allowlist = self._textview_field(
            section,
            "Shell allowlist",
            "One command per line. Only listed commands may run when shell is restricted.",
            height=120,
        )
        self._allowlist.get_buffer().connect("changed", lambda *_: self._save_allowlist())

        self._read_roots = self._textview_field(
            section,
            "Readable directories",
            "One path per line; ~ is expanded. Shell commands may only read files "
            "inside these. Empty means the Keylane install directory only.",
        )
        self._read_roots.get_buffer().connect("changed", lambda *_: self._save_read_roots())

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
            "A local command over stdio, or an HTTP endpoint with a token — "
            "Mailspring's built-in server is one of those. Tools appear as "
            "mcp.<id>.<tool_name>.",
        )

        self._mcp_list = Gtk.ListBox()
        self._mcp_list.add_css_class("settings-item-list")
        self._mcp_list.set_selection_mode(Gtk.SelectionMode.NONE)
        section.append(self._mcp_list)

        form = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        form.add_css_class("settings-mcp-form")

        self._mcp_id = self._entry("server-id")
        self._field(form, "Server ID", self._mcp_id)

        (
            transport_trigger,
            self._mcp_transport_label,
            self._mcp_transport_popover,
            transport_menu,
        ) = self._make_dropdown()
        self._mcp_transport_options: dict[str, Gtk.Button] = {}
        for label, value in _MCP_TRANSPORTS:
            option = Gtk.Button()
            option.add_css_class("settings-dropdown-option")
            option.set_hexpand(True)
            option.set_halign(Gtk.Align.FILL)
            option.set_child(Gtk.Label(label=label, xalign=0, hexpand=True))
            option.connect("clicked", lambda _b, v=value: self._pick_mcp_transport(v))
            transport_menu.append(option)
            self._mcp_transport_options[value] = option
        self._field(form, "Transport", transport_trigger)

        self._mcp_stdio_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._mcp_cmd = self._entry("npx")
        self._field(self._mcp_stdio_box, "Command", self._mcp_cmd)
        self._mcp_args = self._entry("-y @modelcontextprotocol/server-filesystem /home/user")
        self._field(
            self._mcp_stdio_box,
            "Arguments",
            self._mcp_args,
            "Split like a shell line — quote anything with spaces in it. "
            "A comma-separated list still works.",
        )
        self._mcp_env = self._textview_field(
            self._mcp_stdio_box,
            "Environment",
            "Optional. One KEY=value per line, passed to the server process.",
            height=80,
        )
        form.append(self._mcp_stdio_box)

        self._mcp_http_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._mcp_url = self._entry("http://127.0.0.1:2587/mcp")
        self._field(self._mcp_http_box, "Server URL", self._mcp_url)
        self._mcp_token = Gtk.PasswordEntry(show_peek_icon=True)
        self._mcp_token.add_css_class("settings-entry")
        self._mcp_token.set_property("placeholder-text", "token")
        self._field(
            self._mcp_http_box,
            "Bearer token",
            self._mcp_token,
            "Mailspring: Preferences → MCP Server. Paste the bare token — "
            "Keylane adds the \u201cBearer\u201d scheme itself.",
        )
        self._mcp_headers = self._textview_field(
            self._mcp_http_box,
            "Extra headers",
            "Optional. One Name: value per line, sent with every request.",
            height=80,
        )
        form.append(self._mcp_http_box)

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
        self._pick_mcp_transport(self._mcp_transport)

    def _pick_mcp_transport(self, value: str) -> None:
        """Show only the fields that transport actually uses."""
        self._mcp_transport = value
        self._mcp_transport_popover.popdown()
        for label, option_value in _MCP_TRANSPORTS:
            option = self._mcp_transport_options[option_value]
            if option_value == value:
                option.add_css_class("selected-default")
                self._mcp_transport_label.set_text(label)
            else:
                option.remove_css_class("selected-default")
        self._mcp_stdio_box.set_visible(value != "http")
        self._mcp_http_box.set_visible(value == "http")

    def _load_mcp_servers(self) -> None:
        def _work() -> list[dict[str, Any]]:
            return api.get("/mcp/servers", timeout=15).json().get("servers", [])

        self._fetch_async("mcp", _work, self._apply_mcp_servers)

    def _apply_mcp_servers(
        self,
        servers: list[dict[str, Any]] | None,
        error: Exception | None = None,
    ) -> None:
        if error is not None or servers is None:
            self._toast(str(error) if error else "Could not reach the daemon")
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

        top.append(self._badge(server_transport(srv), "muted"))
        source = srv.get("source", "config")
        top.append(self._badge(source, "muted"))

        if source == "user":
            rm = self._secondary_btn("Remove")
            rm.connect("clicked", lambda *_b, s=sid: self._remove_mcp_server(s))
            top.append(rm)

        box.append(top)
        endpoint = server_endpoint(srv)
        if endpoint:
            box.append(
                Gtk.Label(label=endpoint, xalign=0, wrap=True, css_classes=["settings-field-hint"])
            )
        error = str(health.get("error", "")).strip()
        if error and not health.get("ok"):
            # A bad token reads as "offline" otherwise, with nothing to act on.
            box.append(
                Gtk.Label(
                    label=error[:160],
                    xalign=0,
                    wrap=True,
                    css_classes=["settings-field-hint"],
                )
            )
        row.set_child(box)
        return row

    def _add_mcp_server(self, *_args) -> None:
        payload = self._mcp_payload()
        if payload is None:
            return

        def _work() -> None:
            try:
                api.post("/mcp/servers", json=payload, timeout=60).raise_for_status()
                GLib.idle_add(self._toast, "MCP server added")
                GLib.idle_add(self._load_mcp_servers)
                GLib.idle_add(self._clear_mcp_form)
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self._toast, self._request_error(exc))

        threading.Thread(target=_work, daemon=True).start()

    def _mcp_payload(self) -> dict[str, Any] | None:
        """The server as typed, or None with a toast saying what is missing."""
        sid = self._mcp_id.get_text().strip()
        if not sid:
            self._toast("Server ID required")
            return None

        if self._mcp_transport == "http":
            url = self._mcp_url.get_text().strip()
            if not url:
                self._toast("URL required")
                return None
            payload: dict[str, Any] = {"id": sid, "transport": "http", "url": url}
            token = self._mcp_token.get_text().strip()
            if token:
                payload["auth_header"] = token
            headers = parse_header_lines(self._lines(self._mcp_headers))
            if headers:
                payload["headers"] = headers
            return payload

        command = self._mcp_cmd.get_text().strip()
        if not command:
            self._toast("Command required")
            return None
        payload = {
            "id": sid,
            "transport": "stdio",
            "command": command,
            "args": parse_args_field(self._mcp_args.get_text()),
        }
        env = parse_env_lines(self._lines(self._mcp_env))
        if env:
            payload["env"] = env
        return payload

    def _clear_mcp_form(self) -> None:
        for entry in (self._mcp_id, self._mcp_cmd, self._mcp_args, self._mcp_url, self._mcp_token):
            entry.set_text("")
        for view in (self._mcp_env, self._mcp_headers):
            view.get_buffer().set_text("")

    @staticmethod
    def _request_error(exc: Exception) -> str:
        """Prefer the daemon's own message; a raised status only says 400."""
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                detail = response.json().get("detail")
            except Exception:  # noqa: BLE001
                detail = None
            if detail:
                return str(detail)
        return str(exc)

    def _remove_mcp_server(self, server_id: str) -> None:
        def _work() -> None:
            try:
                api.delete(f"/mcp/servers/{server_id}", timeout=30).raise_for_status()
                GLib.idle_add(self._toast, "Removed")
                GLib.idle_add(self._load_mcp_servers)
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self._toast, str(exc))

        threading.Thread(target=_work, daemon=True).start()

    def _reload_mcp(self, *_args) -> None:
        def _work() -> None:
            try:
                r = api.post("/mcp/reload", timeout=60).json()
                GLib.idle_add(self._toast, f"Loaded {r.get('tools_loaded', 0)} MCP tools")
                GLib.idle_add(self._load_mcp_servers)
            except Exception as exc:  # noqa: BLE001
                GLib.idle_add(self._toast, str(exc))

        threading.Thread(target=_work, daemon=True).start()

    @staticmethod
    def _lines(view: Gtk.TextView) -> list[str]:
        buf = view.get_buffer()
        start, end = buf.get_bounds()
        return [ln.strip() for ln in buf.get_text(start, end, False).splitlines() if ln.strip()]

    def _save_allowlist(self) -> None:
        if self._block_save:
            return
        self._patch("security", {"shell_allowlist": self._lines(self._allowlist)})

    def _save_read_roots(self) -> None:
        if self._block_save:
            return
        self._patch("security", {"shell_read_roots": self._lines(self._read_roots)})

    def _test_searx(self, *_args) -> None:
        try:
            r = api.get("/research/health", timeout=15).json()
            ok = r.get("searxng", {}).get("ok", False)
            self._toast("SearXNG OK" if ok else "SearXNG failed")
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc))

    def _test_tts(self, *_args) -> None:
        try:
            api.post("/settings/test/tts", timeout=30)
            self._toast("TTS test sent")
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc))

    def _test_notify(self, *_args) -> None:
        try:
            api.post("/settings/test/notification", timeout=10)
            self._toast("Notification sent")
        except Exception as exc:  # noqa: BLE001
            self._toast(str(exc))

    def load_settings(self) -> None:
        """Fill the panel from the daemon, without blocking the window opening.

        The window is presented first and populates a moment later. Fetching
        first meant every click on the gear froze the UI until the daemon
        answered — and if the daemon was down, for the whole five-second
        timeout, which reads as a button that does nothing.
        """
        self._block_save = True

        def _work() -> dict[str, Any]:
            return api.get("/settings", timeout=5).json()

        def _apply(data: dict[str, Any] | None, error: Exception | None) -> None:
            if error is not None or data is None:
                self._block_save = False
                self._toast(f"Cannot reach the daemon: {error}" if error else "No settings")
                return
            self._apply_settings(data)

        self._fetch_async("settings", _work, _apply)

    def _apply_settings(self, data: dict[str, Any]) -> None:
        assistant = data.get("assistant", {})
        self._name_entry.set_text(str(assistant.get("name", "Keylane")))
        self._user_name_entry.set_text(str(assistant.get("user_name", "") or ""))
        self._budget_spin.set_value(int(assistant.get("iteration_budget", 12)))
        self._auto_learn.set_active(bool(assistant.get("auto_learn_skills", False)))

        ui = data.get("ui", {})
        self._show_theme(str(ui.get("theme_id", "") or "glass-console"))
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
        roots = security.get("shell_read_roots", []) or []
        self._read_roots.get_buffer().set_text("\n".join(str(r) for r in roots))

        models_cfg = data.get("models", {})
        devices = models_cfg.get("devices", {})
        self._model_devices = {
            str(k): str(v) for k, v in devices.items() if isinstance(devices, dict)
        }
        runtime_id = str(models_cfg.get("runtime", "openvino") or "openvino")
        if runtime_id in self._runtime_buttons:
            self._runtime_id = runtime_id
            self._runtime_buttons[runtime_id].set_active(True)
        self._adapters = list(models_cfg.get("adapters", []) or [])
        gpu = next((a for a in self._adapters if a.get("id") == "gpu"), {})
        self._gpu_enabled.set_active(bool(gpu.get("enabled", False)))
        self._gpu_url.set_text(str(gpu.get("base_url", "") or ""))
        self._gpu_unload.set_active(bool(gpu.get("auto_unload", False)))
        self._gpu_model_id = str(gpu.get("model", "") or "")
        self._gpu_model_label.set_text(self._gpu_model_id or "Connect to see models")

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
        self._load_routes()
        self._load_mcp_servers()
