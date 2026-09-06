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
# The MiniCPM5 dialect, detected by its attribute rather than its tag.
_FUNCTION_NAME_HINT = re.compile(r"<function\s+name\s*=", re.IGNORECASE)


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

    parsed = _parse_function_xml(text)
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
    if f"<{_TOOL_TAG}" in lower or "<function_call>" in lower:
        return True
    # `<function name=` rather than `<function`: prose about a function should
    # not be mistaken for a call, and the attribute is what makes it one.
    return bool(_FUNCTION_NAME_HINT.search(text))


def strip_tool_call(text: str) -> str:
    text = _TOOL_CLOSED.sub("", text)
    text = re.sub(rf"<{_TOOL_TAG}>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = _FUNCTION_CLOSED.sub("", text)
    text = re.sub(r"<function_call>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = _FUNCTION_XML.sub("", text)
    text = _FUNCTION_XML_EMPTY.sub("", text)
    # An unterminated call: the stream was cut mid-markup, and the tail is not
    # something to show anyone.
    text = re.sub(
        r"<function\s+name\s*=.*", "", text, flags=re.DOTALL | re.IGNORECASE
    )
    return text.strip()


# ── the XML-parameter dialect ────────────────────────────────────────────
#
# MiniCPM5 and the models that follow its template do not emit JSON. Their
# chat template trains them to write:
#
#     <function name="web_search"><param name="question">weather</param></function>
#
# and to wrap any value containing `<`, `&` or a newline in CDATA. A model
# emitting that into a parser that only knows `<tool_call>{...}</tool_call>`
# does not fail loudly — every branch misses, the call falls through as prose,
# and the model looks as though it simply declined to use its tools. That is
# the whole reason this exists: on a 1B model the tool call *is* the product.

_FUNCTION_XML = re.compile(
    r"<function\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</function\s*>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_XML = re.compile(
    r"<param\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>(.*?)</param\s*>",
    re.DOTALL | re.IGNORECASE,
)
_CDATA = re.compile(r"^\s*<!\[CDATA\[(.*?)\]\]>\s*$", re.DOTALL)

# Bare `<function name="x"/>` — a call with no arguments at all.
_FUNCTION_XML_EMPTY = re.compile(
    r"<function\s+name\s*=\s*[\"']([^\"']+)[\"']\s*/>",
    re.IGNORECASE,
)


def _coerce(value: str) -> Any:
    """An XML parameter value as the type the schema probably wants.

    XML carries no types, so `"seconds": "4"` arrives as a string where the
    schema says number. JSON is tried first because that covers numbers,
    booleans, arrays and objects in one rule — and anything it rejects is
    prose, which was a string to begin with.
    """
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    if text[0] in "[{-0123456789" or text[0].isdigit():
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return value.strip()


def _parse_function_xml(text: str) -> dict[str, Any] | None:
    """The first `<function name=…>` call in *text*, or None."""
    match = _FUNCTION_XML.search(text)
    if match is None:
        empty = _FUNCTION_XML_EMPTY.search(text)
        if empty is None:
            return None
        return {"name": empty.group(1).strip(), "arguments": {}}

    name = match.group(1).strip()
    if not name:
        return None

    arguments: dict[str, Any] = {}
    for param in _PARAM_XML.finditer(match.group(2)):
        key = param.group(1).strip()
        if not key:
            continue
        raw = param.group(2)
        cdata = _CDATA.match(raw)
        # CDATA is verbatim by definition: a value wrapped in it was wrapped
        # because it contains markup or newlines, so it must not be coerced.
        arguments[key] = cdata.group(1) if cdata else _coerce(raw)
    return {"name": name, "arguments": arguments}
