"""MCP client helpers (stdio + Streamable HTTP) for MCP-backed plugins."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class McpError(RuntimeError):
    pass


def normalize_auth_header(value: str | None) -> str:
    """Ensure ``Authorization`` values carry a Bearer scheme when needed.

    Mailspring (and most MCP HTTP servers) reject a bare token. Users often
    paste only the UUID from Preferences → MCP Server.
    """
    text = (value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("bearer ") or lowered.startswith("basic "):
        return text
    # Already a full header value like "Authorization: Bearer …"
    if lowered.startswith("authorization:"):
        return text.split(":", 1)[1].strip() or text
    return f"Bearer {text}"


def _merged_env(extra: dict[str, str] | None) -> dict[str, str] | None:
    """Merge extra vars into the current process env.

    StdioServerParameters replaces the whole environment when ``env`` is set,
    so we must start from ``os.environ`` (PATH, HOME, COMFY_*, etc.).
    """
    if not extra:
        return None
    merged = {str(k): str(v) for k, v in os.environ.items()}
    for key, value in extra.items():
        text = str(value)
        if str(key).upper() in {"AUTH_HEADER", "AUTHORIZATION", "MCP_AUTH"}:
            text = normalize_auth_header(text)
        merged[str(key)] = text
    return merged


@asynccontextmanager
async def mcp_stdio_session(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> AsyncIterator[Any]:
    """Open a short-lived MCP ClientSession over stdio."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:  # noqa: BLE001
        raise McpError(
            "The 'mcp' Python package is required for MCP plugins. "
            "Install with: pip install mcp"
        ) from exc

    params = StdioServerParameters(
        command=command,
        args=args or [],
        env=_merged_env(env),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


@asynccontextmanager
async def mcp_http_session(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> AsyncIterator[Any]:
    """Open a short-lived MCP ClientSession over Streamable HTTP."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        from mcp.shared._httpx_utils import create_mcp_http_client
    except ImportError as exc:  # noqa: BLE001
        raise McpError(
            "The 'mcp' Python package is required for MCP plugins. "
            "Install with: pip install mcp"
        ) from exc

    cleaned_headers: dict[str, str] = {}
    for key, value in (headers or {}).items():
        if not value:
            continue
        if key.lower() == "authorization":
            cleaned_headers[key] = normalize_auth_header(value)
        else:
            cleaned_headers[key] = str(value)

    http = create_mcp_http_client(headers=cleaned_headers or None)
    async with http:
        async with streamable_http_client(url, http_client=http) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


def _tools_from_session_result(result: Any) -> list[dict[str, Any]]:
    tools = getattr(result, "tools", result) or []
    out: list[dict[str, Any]] = []
    for tool in tools:
        schema = getattr(tool, "inputSchema", None) or getattr(
            tool, "input_schema", None
        )
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        out.append(
            {
                "name": getattr(tool, "name", None),
                "description": getattr(tool, "description", "") or "",
                "input_schema": schema if isinstance(schema, dict) else None,
            }
        )
    return out


async def mcp_list_tools(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    async with mcp_stdio_session(command, args, env) as session:
        result = await session.list_tools()
        return _tools_from_session_result(result)


async def mcp_list_tools_http(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    async with mcp_http_session(url, headers=headers) as session:
        result = await session.list_tools()
        return _tools_from_session_result(result)


async def mcp_call_tool(
    command: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> Any:
    async with mcp_stdio_session(command, args, env) as session:
        result = await session.call_tool(tool, arguments or {})
        return _normalize_tool_result(result)


async def mcp_call_tool_http(
    url: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> Any:
    async with mcp_http_session(url, headers=headers) as session:
        result = await session.call_tool(tool, arguments or {})
        return _normalize_tool_result(result)


def _normalize_tool_result(result: Any) -> Any:
    """Flatten MCP CallToolResult into JSON-friendly data."""
    if result is None:
        return None
    if hasattr(result, "content"):
        parts: list[Any] = []
        for block in result.content or []:
            btype = getattr(block, "type", None)
            if btype == "text" or hasattr(block, "text"):
                parts.append(getattr(block, "text", str(block)))
            elif btype == "image" or hasattr(block, "data"):
                parts.append(
                    {
                        "type": "image",
                        "mimeType": getattr(block, "mimeType", None),
                        "data": getattr(block, "data", None),
                    }
                )
            else:
                parts.append(str(block))
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return {"structured": structured, "content": parts}
        if len(parts) == 1:
            return parts[0]
        return parts
    if isinstance(result, (dict, list, str, int, float, bool)):
        return result
    return str(result)
