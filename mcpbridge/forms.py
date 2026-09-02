"""Turning what a user types in the MCP form into a server entry.

Kept free of GTK and of the SDK so both the settings window and the tests
can import it: the parsing is the part worth pinning down, not the widgets.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from typing import Any

HTTP_TRANSPORTS = {"http", "streamable-http", "streamable_http", "sse"}


def server_transport(srv: dict[str, Any]) -> str:
    """Normalize the transport name; a bare ``url`` implies HTTP."""
    transport = str(srv.get("transport", "")).strip().lower()
    if transport in HTTP_TRANSPORTS:
        return "http"
    if transport == "stdio":
        return "stdio"
    return "http" if srv.get("url") else "stdio"


def parse_args_field(text: str | None) -> list[str]:
    """Split an argument line the way the user most likely meant it.

    Commas win when they are there — the field asked for them for a long
    time — otherwise the line splits like a shell command, so a quoted path
    with spaces survives as one argument.
    """
    line = (text or "").strip()
    if not line:
        return []
    if "," in line:
        return [part.strip() for part in line.split(",") if part.strip()]
    try:
        return shlex.split(line)
    except ValueError:  # an unbalanced quote — take the words as typed
        return line.split()


def parse_env_lines(lines: Iterable[str]) -> dict[str, str]:
    """``KEY=value`` per line, for the environment of a stdio server."""
    out: dict[str, str] = {}
    for line in lines:
        key, sep, value = str(line).partition("=")
        if not sep or not key.strip():
            continue
        out[key.strip()] = value.strip()
    return out


def parse_header_lines(lines: Iterable[str]) -> dict[str, str]:
    """``Name: value`` per line; ``Name=value`` is accepted just as well."""
    out: dict[str, str] = {}
    for line in lines:
        key, sep, value = str(line).partition(":")
        if not sep:
            key, sep, value = str(line).partition("=")
        if not sep or not key.strip() or not value.strip():
            continue
        out[key.strip()] = value.strip()
    return out


def mask_token(value: str | None) -> str:
    """Enough of a token to recognise it, never enough to reuse it."""
    text = (value or "").strip()
    if not text:
        return ""
    if text.lower().startswith(("bearer ", "basic ")):
        text = text.split(" ", 1)[1].strip()
    return f"••••{text[-4:]}" if len(text) > 8 else "••••"


def server_endpoint(srv: dict[str, Any]) -> str:
    """The line under a server's name: where it is and how it is reached."""
    if server_transport(srv) == "http":
        url = str(srv.get("url", "")).strip()
        headers = srv.get("headers") or {}
        auth = srv.get("auth_header") or srv.get("token") or ""
        if not auth:
            auth = next((v for k, v in headers.items() if str(k).lower() == "authorization"), "")
        masked = mask_token(str(auth))
        return f"{url}  ·  {masked}" if masked else url
    command = str(srv.get("command", "")).strip()
    args = [str(a) for a in srv.get("args", [])]
    return shlex.join([command, *args]).strip() if command else " ".join(args)
