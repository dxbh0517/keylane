"""Extract and sanitize user-facing text from model output."""

from __future__ import annotations

import re

# Qwen 3.x reasoning tags (split literals so editors don't strip them).
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"
_THINKING_TAG_PAIRS: tuple[tuple[str, str], ...] = (
    (_THINK_OPEN, _THINK_CLOSE),
    ("<think>", "</think>"),
)
_THINKING_PREFIXES = ("Thinking Process:", "Thinking:")
_TOOL_HOLD_MARKERS = (
    "<tool_call",
    "<function_call",
    "<tool_call>",
    "<function_call>",
    "</think>",
)
_REASONING_LINE = re.compile(
    r"^\s*(?:the user is asking|let me |i should |i'll |i've |since i've|actually,|wait,|now i need)",
    re.IGNORECASE,
)
_SOURCES_SPLIT = re.compile(r"\n\s*Sources\s*\n", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove reasoning/thinking sections from a completed model response."""
    if not text:
        return text

    cleaned = text
    for start_tag, end_tag in _THINKING_TAG_PAIRS:
        while True:
            start = cleaned.find(start_tag)
            if start == -1:
                break
            end = cleaned.find(end_tag, start)
            if end == -1:
                cleaned = cleaned[:start]
                break
            cleaned = cleaned[:start] + cleaned[end + len(end_tag) :]

    for prefix in _THINKING_PREFIXES:
        stripped = cleaned.lstrip()
        if stripped.startswith(prefix):
            remainder = stripped[len(prefix) :].lstrip("\n")
            parts = re.split(r"\n\s*\n", remainder, maxsplit=1)
            cleaned = parts[1] if len(parts) == 2 and parts[1].strip() else ""
            break

    # Orphan closing tags — drop reasoning that appears before the close tag.
    while "</think>" in cleaned:
        idx = cleaned.find("</think>")
        head = cleaned[:idx]
        if "<think>" not in head and _THINK_OPEN not in head:
            cleaned = cleaned[idx + len("</think>") :].lstrip("\n ")
        else:
            cleaned = cleaned[:idx] + cleaned[idx + len("</think>") :]

    while _THINK_CLOSE in cleaned:
        idx = cleaned.find(_THINK_CLOSE)
        head = cleaned[:idx]
        if _THINK_OPEN not in head:
            cleaned = cleaned[idx + len(_THINK_CLOSE) :].lstrip("\n ")
        else:
            cleaned = cleaned[:idx] + cleaned[idx + len(_THINK_CLOSE) :]

    return cleaned.strip()


def _strip_tool_markup(text: str) -> str:
    text = re.sub(
        r"<tool_call>\s*.*?\s*</tool_call>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<tool_call>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<function_call>\s*.*?\s*</function_call>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<function_call>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _strip_reasoning_lines(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if _REASONING_LINE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def sanitize_response(text: str) -> str:
    """Full post-decode cleanup: thinking blocks, then tool markup."""
    cleaned = _strip_tool_markup(strip_thinking(text))
    cleaned = _strip_reasoning_lines(cleaned)
    prev = None
    while cleaned != prev:
        prev = cleaned
        cleaned = _strip_tool_markup(strip_thinking(cleaned))
        cleaned = _strip_reasoning_lines(cleaned)
    return cleaned.strip()


def extract_user_answer(text: str) -> str:
    """Return the best user-facing slice of a model response."""
    direct = sanitize_response(text)
    if direct:
        return direct

    # Last segment after a thinking close tag sometimes holds the real answer.
    for close_tag in ("</think>", _THINK_CLOSE):
        if close_tag in text:
            tail = text.rsplit(close_tag, 1)[-1]
            tail = sanitize_response(tail)
            if tail:
                return tail

    return ""


def split_canvas_sections(text: str) -> tuple[str, str]:
    """Split answer body and Sources footer for canvas layout."""
    parts = _SOURCES_SPLIT.split(text.strip(), maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


class ThinkingStreamFilter:
    """Incremental filter for streamed decode chunks."""

    def __init__(self) -> None:
        self._buffer = ""
        self._in_thinking = False
        self._mode: str | None = None
        self._active_pair: tuple[str, str] | None = None

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buffer += chunk
        return self._drain(emit_partial=True)

    def flush(self) -> str:
        if self._in_thinking:
            self._buffer = ""
            return ""
        emitted = self._buffer
        self._buffer = ""
        return emitted

    def _drain(self, *, emit_partial: bool) -> str:
        out: list[str] = []
        while self._buffer:
            if not self._mode:
                self._detect_mode()
                if not self._mode:
                    if len(self._buffer) < 48:
                        return "".join(out)
                    self._mode = "plain"

            if self._mode == "plain":
                for start_tag, end_tag in _THINKING_TAG_PAIRS:
                    if start_tag in self._buffer:
                        before, _, rest = self._buffer.partition(start_tag)
                        out.append(before)
                        self._buffer = rest
                        self._mode = "tag"
                        self._in_thinking = True
                        self._active_pair = (start_tag, end_tag)
                        break
                else:
                    for prefix in _THINKING_PREFIXES:
                        if self._buffer.startswith(prefix):
                            self._mode = "prefix"
                            self._in_thinking = True
                            break
                    else:
                        if emit_partial:
                            out.append(self._buffer)
                            self._buffer = ""
                        break
                if self._mode != "plain":
                    continue

            if self._mode == "tag" and self._active_pair:
                end_tag = self._active_pair[1]
                end = self._buffer.find(end_tag)
                if end == -1:
                    self._buffer = self._buffer[-4096:]
                    break
                self._buffer = self._buffer[end + len(end_tag) :].lstrip("\n ")
                self._in_thinking = False
                self._mode = "plain"
                self._active_pair = None
                continue

            if self._mode == "prefix":
                split = re.split(r"\n\s*\n", self._buffer, maxsplit=1)
                if len(split) == 1:
                    self._buffer = self._buffer[-4096:]
                    break
                self._buffer = split[1].lstrip("\n ")
                self._in_thinking = False
                self._mode = "plain"
                continue

            if self._mode == "orphan_close":
                self._in_thinking = False
                self._mode = "plain"
                continue

        return "".join(out)

    def _detect_mode(self) -> None:
        sample = self._buffer.lstrip()
        if sample.startswith("</think>"):
            self._mode = "orphan_close"
            self._in_thinking = True
            self._buffer = sample[len("</think>") :].lstrip("\n ")
            return
        for start_tag, end_tag in _THINKING_TAG_PAIRS:
            if sample.startswith(start_tag):
                self._mode = "tag"
                self._in_thinking = True
                self._active_pair = (start_tag, end_tag)
                self._buffer = sample[len(start_tag) :]
                return
        for prefix in _THINKING_PREFIXES:
            if sample.startswith(prefix):
                self._mode = "prefix"
                self._in_thinking = True
                return
        if sample.startswith("<redacted") or sample.startswith("Thinking") or sample.startswith("<" + "think"):
            return
        self._mode = "plain"


class OutputStreamFilter:
    """Streaming filter: thinking blocks + hold tool-call markup from the UI."""

    def __init__(self) -> None:
        self._thinking = ThinkingStreamFilter()
        self._hold = ""
        self._tool_hold = False

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        visible = self._thinking.feed(chunk)
        if not visible:
            return ""
        self._hold += visible
        lower = self._hold.lower()
        if any(marker in lower for marker in _TOOL_HOLD_MARKERS):
            self._tool_hold = True
            return ""
        if self._tool_hold:
            return ""
        emitted = self._hold
        self._hold = ""
        return emitted

    def flush(self) -> str:
        tail = self._thinking.flush()
        if self._tool_hold:
            self._hold = ""
            self._tool_hold = False
            return ""
        if tail:
            self._hold += tail
        emitted = self._hold
        self._hold = ""
        return emitted
