"""Render a model answer as a formatted canvas rather than one text blob.

The HUD is narrow, so an answer has to be *scannable*: a headline you can read
at a glance, then structure. This turns the model's markdown into a list of
typed blocks that ui/main.py builds real widgets from — headings, bullets,
key/value rows, code, and quotes each get their own treatment instead of all
arriving as one wrapped paragraph.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["headline", "heading", "text", "bullets", "numbers", "code", "quote", "rule", "kv"]

# Citation markers are stripped: the HUD no longer shows a source list, so a
# dangling "[2]" points at nothing.
# Inline only: a marker at the start of a line is structure (the research
# fallback numbers its excerpts that way), not a citation on a sentence.
_CITATION = re.compile(r"(?<=\S)[ \t]*\[\d+\](?=[\s.,;:)]|$)")
# Matches "Sources", "## Sources", "**Sources:**", "**Sources**:" and friends.
_SOURCES_HEADING = re.compile(r"^[\s#*_]*sources?[\s:*_]*$", re.IGNORECASE)
# A *source entry* carries a URL: "[1] Title — https://…". A line like
# "[1] some quoted text…" is body content from the research fallback and must
# survive — stripping those emptied the answer card completely.
_SOURCE_ENTRY = re.compile(r"^\s*\[\d+\]\s+.*https?://", re.IGNORECASE)
_NUMBERED_SNIPPET = re.compile(r"^\s*\[(\d+)\]\s+(.+)$")
_BASED_ON = re.compile(r"^\s*based on \d+ sources?:?\s*$", re.IGNORECASE)

_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_NUMBER = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_HEADING = re.compile(r"^\s*(#{1,6})\s+(.*)$")
_BOLD_HEADING = re.compile(r"^\s*\*\*(.+?)\*\*:?\s*$")
_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_FENCE = re.compile(r"^\s*```(\w*)\s*$")
# "Term — definition" / "Term: definition" rows, common in model answers.
_KV = re.compile(r"^\s*[-*•]\s+\*\*(?P<key>[^*]{1,40})\*\*\s*(?:[:—–-]\s*)(?P<val>.+)$")


@dataclass
class Block:
    kind: BlockKind
    text: str = ""
    items: list[str] = field(default_factory=list)
    pairs: list[tuple[str, str]] = field(default_factory=list)
    language: str = ""


def strip_sources(text: str) -> str:
    """Drop a trailing Sources section, source entries, and citation markers."""
    kept: list[str] = []
    for line in text.splitlines():
        if _SOURCES_HEADING.match(line):
            break  # everything after a Sources heading is the list itself
        kept.append(line)

    body = [ln for ln in kept if not _SOURCE_ENTRY.match(ln) and not _BASED_ON.match(ln)]
    cleaned = _CITATION.sub("", "\n".join(body)).strip()
    if cleaned:
        return cleaned
    # Never blank the card just because every line looked like a source entry.
    return _CITATION.sub("", "\n".join(kept)).strip()


def _inline(text: str) -> str:
    """Markdown inline spans → Pango markup, on already-escaped text."""
    out = html.escape(text.strip())
    out = re.sub(r"`([^`]+)`", r'<span font_family="monospace" size="93%">\1</span>', out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![\w*])\*([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", out)
    out = re.sub(r"(?<![\w_])_([^_\n]+?)_(?![\w_])", r"<i>\1</i>", out)
    # Show link text, not the URL — there is no room for one in the HUD.
    out = re.sub(r"\[([^\]]+)\]\((?:[^)]+)\)", r"\1", out)
    return out


def _flush(buf: list[str], blocks: list[Block], kind: BlockKind = "text") -> None:
    if not buf:
        return
    joined = " ".join(part.strip() for part in buf if part.strip()).strip()
    if joined:
        blocks.append(Block(kind, text=joined))
    buf.clear()


def parse_blocks(answer: str) -> list[Block]:
    """Split an answer into renderable blocks. The first line becomes a headline."""
    text = strip_sources(answer)
    if not text:
        return []

    blocks: list[Block] = []
    para: list[str] = []
    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        fence = _FENCE.match(line)
        if fence:
            _flush(para, blocks)
            language = fence.group(1)
            i += 1
            code: list[str] = []
            while i < len(lines) and not _FENCE.match(lines[i]):
                code.append(lines[i])
                i += 1
            i += 1
            blocks.append(Block("code", text="\n".join(code).rstrip(), language=language))
            continue

        if not line.strip():
            _flush(para, blocks)
            i += 1
            continue

        if _RULE.match(line):
            _flush(para, blocks)
            blocks.append(Block("rule"))
            i += 1
            continue

        heading = _HEADING.match(line) or _BOLD_HEADING.match(line)
        if heading:
            _flush(para, blocks)
            blocks.append(Block("heading", text=heading.groups()[-1].strip()))
            i += 1
            continue

        quote = _QUOTE.match(line)
        if quote:
            _flush(para, blocks)
            body = [quote.group(1)]
            i += 1
            while i < len(lines) and (q := _QUOTE.match(lines[i])):
                body.append(q.group(1))
                i += 1
            blocks.append(Block("quote", text=" ".join(b.strip() for b in body).strip()))
            continue

        if _BULLET.match(line):
            _flush(para, blocks)
            rows: list[tuple[str, str] | str] = []
            while i < len(lines) and (bullet := _BULLET.match(lines[i])):
                kv = _KV.match(lines[i])
                rows.append((kv.group("key").strip(), kv.group("val").strip()) if kv else bullet.group(1).strip())
                i += 1
            # A run of "**Term** — value" rows reads far better as a definition
            # list; a mixed list stays a plain list, in the order written.
            if rows and all(isinstance(r, tuple) for r in rows):
                blocks.append(Block("kv", pairs=rows))  # type: ignore[arg-type]
            else:
                blocks.append(
                    Block(
                        "bullets",
                        items=[f"**{r[0]}** — {r[1]}" if isinstance(r, tuple) else r for r in rows],
                    )
                )
            continue

        if _NUMBERED_SNIPPET.match(line):
            # The research fallback answers as "[1] excerpt… / [2] excerpt…".
            # Rendered as a list it is at least readable.
            _flush(para, blocks)
            items = []
            while i < len(lines) and (snip := _NUMBERED_SNIPPET.match(lines[i])):
                items.append(snip.group(2).strip())
                i += 1
            blocks.append(Block("bullets", items=items))
            continue

        if _NUMBER.match(line):
            _flush(para, blocks)
            items = []
            while i < len(lines) and (n := _NUMBER.match(lines[i])):
                items.append(n.group(2).strip())
                i += 1
            blocks.append(Block("numbers", items=items))
            continue

        para.append(line)
        i += 1

    _flush(para, blocks)

    # Promote the opening paragraph to a headline so the answer has a lede.
    for block in blocks:
        if block.kind == "text":
            block.kind = "headline"
            break
        if block.kind in ("bullets", "numbers", "kv", "code", "quote"):
            break

    return blocks


def block_markup(block: Block) -> str:
    """Pango markup for a text-bearing block."""
    if block.kind in ("bullets", "numbers", "kv", "rule", "code"):
        return ""
    return _inline(block.text)


SUMMARY_CHARS = 240
MIN_SENTENCE_CHARS = 90


def trim_to_sentence(text: str, max_chars: int = SUMMARY_CHARS) -> str:
    """Cut to a sentence end within *max_chars*, else to a word boundary."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    # Only honour a sentence end that keeps a useful amount of the text;
    # otherwise a short opening sentence would throw the rest away.
    if cut >= min(MIN_SENTENCE_CHARS, max_chars // 2):
        return window[: cut + 1]
    return window.rsplit(" ", 1)[0].rstrip(",;:—- ") + "…"


def headline_text(answer: str, max_chars: int = SUMMARY_CHARS) -> str:
    """Short gist for the collapsed card, notifications, and titles."""
    for block in parse_blocks(answer):
        if block.kind in ("headline", "heading", "text"):
            return trim_to_sentence(re.sub(r"<[^>]+>", "", block_markup(block)), max_chars)
        if block.kind in ("bullets", "numbers") and block.items:
            return trim_to_sentence(block.items[0], max_chars)
        if block.kind == "kv" and block.pairs:
            key, val = block.pairs[0]
            return trim_to_sentence(f"{key} — {val}", max_chars)
        if block.kind == "code":
            return trim_to_sentence(block.text, max_chars)
    return ""


def is_compact(answer: str, max_chars: int = SUMMARY_CHARS) -> bool:
    """True when the answer is short enough to show whole, with no expander."""
    body = strip_sources(answer)
    blocks = parse_blocks(body)
    if len(blocks) > 1:
        return False
    if len(blocks) == 1 and len(blocks[0].items) > 1:
        return False
    return len(body) <= max_chars


def plain_text(answer: str) -> str:
    """Clean text for the clipboard: no citations, no Sources block."""
    return strip_sources(answer)
