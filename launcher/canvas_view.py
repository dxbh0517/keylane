"""Render a Keylane Canvas as GTK widgets.

The popup shows answers as laid-out content — stat tiles, tables, callouts —
rather than a paragraph of JSON. This is the GTK half of the renderer; the
control panel has an HTML twin in :mod:`app.canvas`.
"""

from __future__ import annotations

from typing import Any

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import Gtk  # noqa: E402

NOTE_CLASS = {
    "info": "canvas-note",
    "success": "canvas-note success",
    "warning": "canvas-note warning",
    "danger": "canvas-note danger",
}


def _label(text: str, *css: str, wrap: bool = True, selectable: bool = True) -> Gtk.Label:
    label = Gtk.Label(label=text)
    label.set_xalign(0.0)
    label.set_wrap(wrap)
    label.set_selectable(selectable)
    if wrap:
        label.set_wrap_mode(2)  # WORD_CHAR
    for name in css:
        label.add_css_class(name)
    return label


def _stats(block: dict[str, Any]) -> Gtk.Widget:
    flow = Gtk.FlowBox()
    flow.set_selection_mode(Gtk.SelectionMode.NONE)
    flow.set_max_children_per_line(4)
    flow.set_column_spacing(8)
    flow.set_row_spacing(8)
    flow.set_homogeneous(True)
    for item in block.get("items") or []:
        tile = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        tile.add_css_class("canvas-stat")
        tile.append(_label(str(item.get("label") or ""), "canvas-stat-label", selectable=False))
        tile.append(_label(str(item.get("value") or ""), "canvas-stat-value"))
        detail = str(item.get("detail") or "")
        if detail:
            tile.append(_label(detail, "canvas-stat-detail", selectable=False))
        flow.append(tile)
    return flow


def _table(block: dict[str, Any]) -> Gtk.Widget:
    grid = Gtk.Grid()
    grid.add_css_class("canvas-table")
    grid.set_column_spacing(14)
    grid.set_row_spacing(4)

    columns = [str(c) for c in (block.get("columns") or [])]
    row_index = 0
    if columns:
        for col, name in enumerate(columns):
            grid.attach(_label(name, "canvas-th", wrap=False, selectable=False), col, 0, 1, 1)
        row_index = 1

    for row in block.get("rows") or []:
        cells = [str(c) for c in row]
        for col, cell in enumerate(cells):
            grid.attach(_label(cell, "canvas-td", wrap=False), col, row_index, 1, 1)
        row_index += 1

    scroller = Gtk.ScrolledWindow()
    scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
    scroller.set_propagate_natural_width(True)
    scroller.set_propagate_natural_height(True)
    scroller.set_child(grid)
    return scroller


def _code(block: dict[str, Any]) -> Gtk.Widget:
    label = _label(str(block.get("text") or ""), "canvas-code", wrap=False)
    scroller = Gtk.ScrolledWindow()
    scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.NEVER)
    scroller.set_propagate_natural_width(True)
    scroller.set_propagate_natural_height(True)
    scroller.set_max_content_height(260)
    scroller.set_child(label)
    scroller.add_css_class("canvas-code-view")
    return scroller


def _list(block: dict[str, Any]) -> Gtk.Widget:
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
    ordered = bool(block.get("ordered"))
    for index, entry in enumerate(block.get("entries") or [], start=1):
        marker = f"{index}." if ordered else "•"
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.append(_label(marker, "canvas-bullet", wrap=False, selectable=False))
        row.append(_label(str(entry), "canvas-text"))
        box.append(row)
    return box


def _links(block: dict[str, Any], on_open) -> Gtk.Widget:
    box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
    for link in block.get("links") or []:
        href = str(link.get("href") or "")
        label = str(link.get("label") or href)
        button = Gtk.Button(label=label)
        button.add_css_class("canvas-link")
        button.set_halign(Gtk.Align.START)
        button.set_has_frame(False)
        button.set_tooltip_text(href)
        if on_open is not None:
            button.connect("clicked", lambda _b, target=href: on_open(target))
        box.append(button)
    return box


def build_canvas(canvas: dict[str, Any] | None, *, on_open=None) -> Gtk.Widget | None:
    """Build the widget tree for a canvas, or ``None`` when there is nothing."""
    if not canvas:
        return None
    blocks = canvas.get("blocks") or []
    title = str(canvas.get("title") or "")
    summary = str(canvas.get("summary") or "")
    if not (blocks or title or summary):
        return None

    root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    root.add_css_class("canvas")

    if title:
        root.append(_label(title, "canvas-title", selectable=False))
    if summary:
        root.append(_label(summary, "canvas-summary"))

    for block in blocks:
        kind = str(block.get("type") or "text")
        if kind == "heading":
            root.append(_label(str(block.get("text") or ""), "canvas-heading", selectable=False))
        elif kind == "text":
            root.append(_label(str(block.get("text") or ""), "canvas-text"))
        elif kind == "note":
            note = _label(str(block.get("text") or ""), "canvas-text")
            wrapper = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            for name in NOTE_CLASS.get(str(block.get("style") or "info"), "canvas-note").split():
                wrapper.add_css_class(name)
            wrapper.append(note)
            root.append(wrapper)
        elif kind == "code":
            root.append(_code(block))
        elif kind == "stats":
            root.append(_stats(block))
        elif kind == "table":
            root.append(_table(block))
        elif kind == "list":
            root.append(_list(block))
        elif kind == "links":
            root.append(_links(block, on_open))

    source = str(canvas.get("source") or "")
    if source:
        root.append(_label(source, "canvas-source", selectable=False))
    return root
