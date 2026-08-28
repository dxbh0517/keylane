"""Parse tool calls from model text output."""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_BLOCK_CLOSED = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_BLOCK_OPEN = re.compile(r"<tool_call>\s*(\{.*)", re.DOTALL | re.IGNORECASE)


def _extract_json_object(text: str) -> str | None:
    text = text.strip()
    if not text.startswith("{"):
        return None
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[: i + 1]
    return None


def _parse_payload(raw_json: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "name" not in data:
        return None
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": str(data["name"]), "arguments": arguments}


def parse_tool_call(text: str) -> dict[str, Any] | None:
    match = _TOOL_BLOCK_CLOSED.search(text)
    if match:
        return _parse_payload(match.group(1))

    match = _TOOL_BLOCK_OPEN.search(text)
    if match:
        blob = _extract_json_object(match.group(1))
        if blob:
            return _parse_payload(blob)

    # Qwen / OpenAI-style function call JSON (whole line or embedded)
    for blob in _find_json_objects(text):
        parsed = _parse_payload(blob)
        if parsed:
            return parsed

    # Qwen tool_call XML variant
    fn_match = re.search(
        r'<function_call>\s*(\{.*?\})\s*</function_call>',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if fn_match:
        return _parse_payload(fn_match.group(1))

    return None


def _find_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        blob = _extract_json_object(text[i:])
        if blob:
            objects.append(blob)
            i += len(blob)
        else:
            i += 1
    return objects


def has_tool_call_markup(text: str) -> bool:
    return "<tool_call>" in text.lower()


def strip_tool_call(text: str) -> str:
    text = _TOOL_BLOCK_CLOSED.sub("", text)
    text = re.sub(r"<tool_call>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()
