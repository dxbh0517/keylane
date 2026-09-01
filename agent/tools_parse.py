"""Parse tool calls from model text output."""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_TAG = "tool_call"
_TOOL_CLOSED = re.compile(
    rf"<{_TOOL_TAG}>\s*(.*?)\s*</{_TOOL_TAG}>",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_OPEN = re.compile(rf"<{_TOOL_TAG}>\s*(.*)", re.DOTALL | re.IGNORECASE)
_FUNCTION_CLOSED = re.compile(
    r"<function_call>\s*(.*?)\s*</function_call>",
    re.DOTALL | re.IGNORECASE,
)
_FUNCTION_OPEN = re.compile(r"<function_call>\s*(.*)", re.DOTALL | re.IGNORECASE)


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
    blob = _extract_json_object(raw_json.strip()) or raw_json.strip()
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "name" not in data:
        return None
    arguments = data.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": str(data["name"]), "arguments": arguments}


def _payload_from_inner(inner: str) -> dict[str, Any] | None:
    blob = _extract_json_object(inner.strip())
    if not blob:
        return None
    return _parse_payload(blob)


def parse_tool_call(text: str) -> dict[str, Any] | None:
    match = _TOOL_CLOSED.search(text)
    if match:
        parsed = _payload_from_inner(match.group(1))
        if parsed:
            return parsed

    match = _TOOL_OPEN.search(text)
    if match:
        parsed = _payload_from_inner(match.group(1))
        if parsed:
            return parsed

    match = _FUNCTION_CLOSED.search(text)
    if match:
        parsed = _payload_from_inner(match.group(1))
        if parsed:
            return parsed

    match = _FUNCTION_OPEN.search(text)
    if match:
        parsed = _payload_from_inner(match.group(1))
        if parsed:
            return parsed

    for blob in _find_json_objects(text):
        parsed = _parse_payload(blob)
        if parsed:
            return parsed

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
    lower = text.lower()
    return f"<{_TOOL_TAG}" in lower or "<function_call>" in lower


def strip_tool_call(text: str) -> str:
    text = _TOOL_CLOSED.sub("", text)
    text = re.sub(rf"<{_TOOL_TAG}>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = _FUNCTION_CLOSED.sub("", text)
    text = re.sub(r"<function_call>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()
