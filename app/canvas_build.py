"""Build a canvas from text, structurally.

Asking a 1.5B model to emit a canvas as JSON does not work reliably — it
loops trying more tools, or returns prose. So the canvas is *derived* from
what the tools actually produced, in Python, where the result is deterministic.

Two converters do the work:

``markdown_to_canvas``  turns headings, tables, lists, fences and callouts
                        into real blocks, so an answer written in markdown
                        renders as layout rather than as literal ``##`` and
                        ``|---|`` characters.
``output_to_canvas``    recognises columnar command output — ``df``, ``ls -l``,
                        ``ps``, ``free`` — and lays it out as a table.

Everything here is best-effort: anything unrecognised stays a paragraph or a
code block, which is exactly what it should be.
"""

from __future__ import annotations

import re

from app.canvas import Block, Canvas, Link, Stat

# ---------------------------------------------------------------- markdown

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^```\s*(\w*)\s*$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_ORDERED = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")
_QUOTE = re.compile(r"^>\s?(.*)$")
_KEY_VALUE = re.compile(r"^\s*([A-Za-z][\w ./-]{0,28}):\s+(\S.{0,60})$")
_LINK = re.compile(r"^\s*\[([^\]]+)\]\(([^)]+)\)\s*$")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
_INLINE_CODE = re.compile(r"`([^`]+)`")

# A callout only when the line actually starts with one of these.
_NOTE_WORDS = {
    "note": "info",
    "tip": "success",
    "warning": "warning",
    "caution": "warning",
    "danger": "danger",
    "error": "danger",
}


def _inline(text: str) -> str:
    """Strip inline markup — the renderer styles, it does not parse."""
    out = _BOLD.sub(r"\1", text)
    out = _ITALIC.sub(r"\1", out)
    out = _INLINE_CODE.sub(r"\1", out)
    return out.strip()


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_to_canvas(text: str, *, title: str = "", source: str = "") -> Canvas:
    """Parse the markdown a model actually writes into canvas blocks."""
    lines = (text or "").splitlines()
    blocks: list[Block] = []
    summary = ""
    index = 0
    total = len(lines)

    def flush(buffer: list[str]) -> None:
        body = " ".join(part.strip() for part in buffer if part.strip()).strip()
        if body:
            blocks.append(Block(type="text", text=_inline(body)))

    paragraph: list[str] = []

    while index < total:
        line = lines[index]
        stripped = line.strip()

        # Fenced code
        fence = _FENCE.match(stripped)
        if fence:
            flush(paragraph)
            paragraph = []
            language = fence.group(1)
            index += 1
            body: list[str] = []
            while index < total and not lines[index].strip().startswith("```"):
                body.append(lines[index])
                index += 1
            index += 1
            if body:
                blocks.append(
                    Block(type="code", text="\n".join(body), language=language)
                )
            continue

        if not stripped:
            flush(paragraph)
            paragraph = []
            index += 1
            continue

        # Headings
        heading = _HEADING.match(stripped)
        if heading:
            flush(paragraph)
            paragraph = []
            level = len(heading.group(1))
            body = _inline(heading.group(2))
            # The first H1/H2 becomes the canvas title rather than a block.
            if level <= 2 and not title and not blocks:
                title = body
            else:
                blocks.append(Block(type="heading", text=body, level=min(level, 4)))
            index += 1
            continue

        # Tables
        if "|" in stripped and index + 1 < total and _TABLE_SEP.match(lines[index + 1]):
            flush(paragraph)
            paragraph = []
            columns = _split_row(stripped)
            index += 2
            rows: list[list[str]] = []
            while index < total and "|" in lines[index] and lines[index].strip():
                rows.append([_inline(c) for c in _split_row(lines[index])])
                index += 1
            if rows:
                blocks.append(
                    Block(type="table", columns=[_inline(c) for c in columns], rows=rows)
                )
            continue

        # Blockquote, possibly a callout
        quote = _QUOTE.match(stripped)
        if quote:
            flush(paragraph)
            paragraph = []
            body_parts: list[str] = []
            while index < total and _QUOTE.match(lines[index].strip()):
                match = _QUOTE.match(lines[index].strip())
                body_parts.append(match.group(1) if match else "")
                index += 1
            body = _inline(" ".join(body_parts))
            style = "info"
            marker = re.match(r"^\*{0,2}(\w+)\*{0,2}\s*[:.]?\s*(.*)$", body)
            if marker and marker.group(1).lower() in _NOTE_WORDS:
                style = _NOTE_WORDS[marker.group(1).lower()]
                body = marker.group(2) or body
            if body:
                blocks.append(Block(type="note", text=body, style=style))
            continue

        # Lists — collected together, and promoted to stats or links when they
        # are really one of those in disguise.
        if _BULLET.match(line) or _ORDERED.match(line):
            flush(paragraph)
            paragraph = []
            ordered = bool(_ORDERED.match(line))
            entries: list[str] = []
            while index < total:
                bullet = _ORDERED.match(lines[index]) if ordered else _BULLET.match(lines[index])
                if not bullet:
                    break
                entries.append(_inline(bullet.group(1)))
                index += 1
            blocks.append(_list_block(entries, ordered))
            continue

        paragraph.append(stripped)
        index += 1

    flush(paragraph)

    # A short leading paragraph reads better as the summary line.
    if blocks and blocks[0].type == "text" and len(blocks[0].text) <= 160:
        summary = blocks.pop(0).text

    return Canvas(title=title, summary=summary, blocks=blocks, source=source).cleaned()


def _list_block(entries: list[str], ordered: bool) -> Block:
    """A list of ``label: value`` pairs is really a stats block."""
    if not ordered and 2 <= len(entries) <= 8:
        pairs = [_KEY_VALUE.match(entry) for entry in entries]
        if all(pairs):
            return Block(
                type="stats",
                items=[
                    Stat(label=m.group(1).strip(), value=m.group(2).strip())
                    for m in pairs
                    if m
                ],
            )
        links = [_LINK.match(entry) for entry in entries]
        if all(links):
            return Block(
                type="links",
                links=[Link(label=m.group(1), href=m.group(2)) for m in links if m],
            )
    return Block(type="list", entries=entries, ordered=ordered)


# ------------------------------------------------------------ tool output


def _looks_columnar(lines: list[str]) -> bool:
    """Two or more lines with aligned columns separated by runs of spaces."""
    hits = sum(1 for line in lines if re.search(r"\S {2,}\S", line))
    return len(lines) >= 2 and hits >= max(2, len(lines) // 2)


def output_to_canvas(
    text: str, *, command: str = "", title: str = "", source: str = ""
) -> Canvas:
    """Lay out command output, as a table when it plainly is one."""
    body = (text or "").strip()
    if not body:
        return Canvas(title=title, source=source)

    lines = [line for line in body.splitlines() if line.strip()]

    # A single line is a sentence, not a table.
    if len(lines) == 1:
        return Canvas(
            title=title, blocks=[Block(type="text", text=lines[0])], source=source
        ).cleaned()

    if _looks_columnar(lines) and len(lines) <= 40:
        table = _as_table(lines)
        if table is not None:
            columns, rows = table
            return Canvas(
                title=title,
                blocks=[Block(type="table", columns=columns, rows=rows)],
                source=source,
            ).cleaned()

    return Canvas(
        title=title,
        blocks=[Block(type="code", text=body, language=_language_for(command))],
        source=source,
    ).cleaned()


def _as_table(lines: list[str]) -> tuple[list[str], list[list[str]]] | None:
    """Split aligned output into columns, or ``None`` if it is not a table.

    Splitting on runs of two spaces is not enough: ``df -h`` writes
    ``Use% Mounted on`` with a single space, and ``ls -l`` separates most
    fields by one. So the *data* rows decide the column count, and the header
    is split to match — with the final column keeping whatever is left, which
    is what holds a mount point or a filename containing spaces.
    """
    if len(lines) < 3:
        return None

    def vote(rows: list[list[str]]) -> tuple[int, int]:
        """Most common row width, and how many rows agree with it."""
        counts = [len(row) for row in rows]
        if not counts:
            return 0, 0
        best = max(set(counts), key=counts.count)
        return best, counts.count(best)

    # Try both splits and keep whichever produces a more consistent table.
    by_runs = [re.split(r"\s{2,}", line.strip()) for line in lines[1:]]
    by_space = [line.split() for line in lines[1:]]
    runs_width, runs_agree = vote(by_runs)
    space_width, space_agree = vote(by_space)

    if space_agree > runs_agree or (space_agree == runs_agree and space_width > runs_width):
        rows_raw = by_space
    else:
        rows_raw = by_runs

    widths = [len(row) for row in rows_raw]
    width = max(set(widths), key=widths.count)  # the most common row shape
    if width < 2:
        return None
    if sum(1 for w in widths if w == width) < max(2, len(widths) // 2):
        return None  # too ragged to be a table

    header = lines[0].split()
    if len(header) > width:
        # More headings than data columns: the tail belongs together, as in
        # ls -l where "Jan 5 10:03" is one date field.
        header = header[: width - 1] + [" ".join(header[width - 1 :])]
    elif len(header) < width:
        # Fewer headings than data columns means the *data* has more fields —
        # df's "Use% Mounted on" is two headings the split already found, so
        # the data shape wins and any shortfall is padded, not merged.
        header += [""] * (width - len(header))

    rows: list[list[str]] = []
    for line in lines[1:]:
        cells = re.split(r"\s{2,}", line.strip())
        if len(cells) != width:
            # Fall back to single-space splitting, the same way the width vote
            # did — otherwise header and rows are cut on different rules.
            single = line.split()
            if len(single) >= width:
                cells = single
        if len(cells) > width:
            # Everything past the last column belongs to it.
            cells = cells[: width - 1] + [" ".join(cells[width - 1 :])]
        elif len(cells) < width:
            cells += [""] * (width - len(cells))
        rows.append(cells)

    return header, rows


def _language_for(command: str) -> str:
    name = (command or "").strip().split()[0] if command else ""
    if name in {"git"}:
        return "diff"
    if name in {"python3", "python", "pip"}:
        return "python"
    return "shell" if name else ""
