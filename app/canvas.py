"""Keylane Canvas — the structured document an answer is rendered from.

An assistant that replies with raw JSON or a wall of markdown makes the reader
do the layout work. A canvas is a small declarative document instead: a title
and an ordered list of blocks, each of which knows how it wants to be shown.
The popup renders it as GTK widgets, the control panel as HTML, and neither has
to parse prose.

The schema is deliberately tiny. A 1.5B model has to be able to emit it
correctly on the first try, so there are eight block types, no nesting beyond
one level, and every field is optional except ``type``.

    {
      "title": "Disk usage",
      "summary": "254 GB free of 952 GB.",
      "blocks": [
        {"type": "stats", "items": [{"label": "Free", "value": "254 GB"}]},
        {"type": "table", "columns": ["Mount", "Use%"],
         "rows": [["/", "74%"]]},
        {"type": "note", "style": "warning", "text": "Root is 74% full."}
      ]
    }
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

BlockType = Literal[
    "text",      # a paragraph
    "heading",   # a section break
    "stats",     # label/value pairs, shown as tiles
    "table",     # columns + rows
    "list",      # bullets or steps
    "code",      # preformatted, with a language hint
    "note",      # callout: info | success | warning | danger
    "links",     # named links or file paths
]

NOTE_STYLES = {"info", "success", "warning", "danger"}


class Stat(BaseModel):
    label: str = ""
    value: str = ""
    detail: str = ""


class Link(BaseModel):
    label: str = ""
    href: str = ""


class Block(BaseModel):
    type: BlockType = "text"

    # text / heading / note / code
    text: str = ""
    level: int = 2
    style: str = "info"
    language: str = ""

    # stats
    items: list[Stat] = Field(default_factory=list)

    # table
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    # list
    ordered: bool = False
    entries: list[str] = Field(default_factory=list)

    # links
    links: list[Link] = Field(default_factory=list)

    @field_validator("style", mode="before")
    @classmethod
    def _known_style(cls, value: Any) -> str:
        text = str(value or "info").strip().lower()
        return text if text in NOTE_STYLES else "info"

    @field_validator("rows", mode="before")
    @classmethod
    def _stringify_rows(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return []
        return [
            [str(cell) for cell in row] if isinstance(row, list) else [str(row)]
            for row in value
        ]

    @field_validator("entries", "columns", mode="before")
    @classmethod
    def _stringify_seq(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    def is_empty(self) -> bool:
        """A block with nothing in it should never be rendered."""
        if self.type in {"text", "heading", "note", "code"}:
            return not self.text.strip()
        if self.type == "stats":
            return not self.items
        if self.type == "table":
            return not self.rows
        if self.type == "list":
            return not self.entries
        if self.type == "links":
            return not self.links
        return True


class Canvas(BaseModel):
    title: str = ""
    summary: str = ""
    blocks: list[Block] = Field(default_factory=list)
    source: str = ""
    """Which worker or tool produced this, shown as a footer."""

    def cleaned(self) -> "Canvas":
        """Drop empty blocks — a canvas exists to show real content."""
        return Canvas(
            title=self.title.strip(),
            summary=self.summary.strip(),
            blocks=[b for b in self.blocks if not b.is_empty()],
            source=self.source.strip(),
        )

    def is_empty(self) -> bool:
        return not (self.title or self.summary or self.blocks)

    def to_text(self) -> str:
        """Plain-text fallback, for logs and clients that cannot render."""
        lines: list[str] = []
        if self.title:
            lines.append(self.title)
        if self.summary:
            lines.append(self.summary)
        for block in self.blocks:
            if block.type == "heading":
                lines.append(f"\n{block.text}")
            elif block.type in {"text", "note"}:
                lines.append(block.text)
            elif block.type == "code":
                lines.append(block.text)
            elif block.type == "stats":
                lines += [f"{s.label}: {s.value}" for s in block.items]
            elif block.type == "list":
                lines += [f"  • {entry}" for entry in block.entries]
            elif block.type == "table":
                if block.columns:
                    lines.append("  ".join(block.columns))
                lines += ["  ".join(row) for row in block.rows]
            elif block.type == "links":
                lines += [f"{link.label}: {link.href}" for link in block.links]
        return "\n".join(line for line in lines if line is not None).strip()


# --------------------------------------------------------------- parsing --


FENCE = re.compile(r"```(?:json|canvas)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_canvas(payload: Any) -> Canvas | None:
    """Build a Canvas from whatever the model produced, or ``None``.

    Accepts a dict, a JSON string, or prose with a fenced JSON block. Returns
    ``None`` rather than raising: a malformed canvas must fall back to plain
    text, never break the answer.
    """
    data: Any = payload
    if isinstance(payload, str):
        text = payload.strip()
        match = FENCE.search(text)
        if match:
            text = match.group(1)
        if not text.startswith("{"):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None
    if "blocks" not in data and "title" not in data and "summary" not in data:
        return None

    try:
        canvas = Canvas(**data).cleaned()
    except Exception:  # noqa: BLE001
        return None
    return None if canvas.is_empty() else canvas


def canvas_from_text(text: str, *, title: str = "", source: str = "") -> Canvas:
    """Wrap plain text in a minimal canvas so there is always something to show.

    Long command output and code get a ``code`` block; anything else becomes
    paragraphs. This is the floor, not the goal — the model is asked for a real
    canvas first.
    """
    body = (text or "").strip()
    blocks: list[Block] = []
    if body:
        lines = body.splitlines()
        # Columnar command output (df, ls, ps) is the common case: several
        # lines where values are aligned with runs of spaces.
        looks_preformatted = len(lines) >= 2 and (
            sum(1 for line in lines if re.search(r"\S {2,}\S", line)) >= 2
            or sum(1 for line in lines if line.startswith((" ", "\t", "$", "#"))) >= 2
        )
        if looks_preformatted:
            blocks.append(Block(type="code", text=body))
        else:
            for paragraph in re.split(r"\n\s*\n", body):
                chunk = paragraph.strip()
                if chunk:
                    blocks.append(Block(type="text", text=chunk))
    return Canvas(title=title, blocks=blocks, source=source).cleaned()


# ------------------------------------------------------------ HTML render --


def _esc(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_html(canvas: Canvas) -> str:
    """Render a canvas as HTML for the control panel.

    Class names only — the panel's stylesheet owns the look, so a theme
    restyles canvases along with everything else.
    """
    parts: list[str] = ['<article class="canvas">']
    if canvas.title:
        parts.append(f'<h2 class="canvas-title">{_esc(canvas.title)}</h2>')
    if canvas.summary:
        parts.append(f'<p class="canvas-summary">{_esc(canvas.summary)}</p>')

    for block in canvas.blocks:
        if block.type == "heading":
            level = min(max(block.level, 2), 4)
            parts.append(f"<h{level}>{_esc(block.text)}</h{level}>")
        elif block.type == "text":
            parts.append(f"<p>{_esc(block.text)}</p>")
        elif block.type == "note":
            parts.append(
                f'<div class="canvas-note {_esc(block.style)}">{_esc(block.text)}</div>'
            )
        elif block.type == "code":
            lang = f' data-lang="{_esc(block.language)}"' if block.language else ""
            parts.append(f"<pre{lang}><code>{_esc(block.text)}</code></pre>")
        elif block.type == "stats":
            tiles = "".join(
                f'<div class="canvas-stat"><span class="label">{_esc(s.label)}</span>'
                f'<strong class="value">{_esc(s.value)}</strong>'
                + (f'<span class="detail">{_esc(s.detail)}</span>' if s.detail else "")
                + "</div>"
                for s in block.items
            )
            parts.append(f'<div class="canvas-stats">{tiles}</div>')
        elif block.type == "table":
            head = (
                "<thead><tr>"
                + "".join(f"<th>{_esc(c)}</th>" for c in block.columns)
                + "</tr></thead>"
                if block.columns
                else ""
            )
            body = "".join(
                "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>"
                for row in block.rows
            )
            parts.append(
                f'<div class="canvas-table"><table>{head}<tbody>{body}</tbody></table></div>'
            )
        elif block.type == "list":
            tag = "ol" if block.ordered else "ul"
            items = "".join(f"<li>{_esc(e)}</li>" for e in block.entries)
            parts.append(f"<{tag}>{items}</{tag}>")
        elif block.type == "links":
            items = "".join(
                f'<li><a href="{_esc(link.href)}">{_esc(link.label or link.href)}</a></li>'
                for link in block.links
            )
            parts.append(f'<ul class="canvas-links">{items}</ul>')

    if canvas.source:
        parts.append(f'<p class="canvas-source">{_esc(canvas.source)}</p>')
    parts.append("</article>")
    return "".join(parts)
