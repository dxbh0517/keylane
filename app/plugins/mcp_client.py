"""MCP client helpers (stdio + optional HTTP) for MCP-backed plugins."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class McpError(RuntimeError):
    pass


def _merged_env(extra: dict[str, str] | None) -> dict[str, str] | None:
    """Merge extra vars into the current process env.

    StdioServerParameters replaces the whole environment when ``env`` is set,
    so we must start from ``os.environ`` (PATH, HOME, COMFY_*, etc.).
    """
    if not extra:
        return None
    merged = {str(k): str(v) for k, v in os.environ.items()}
    merged.update({str(k): str(v) for k, v in extra.items()})
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


async def mcp_list_tools(
    command: str,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    async with mcp_stdio_session(command, args, env) as session:
        result = await session.list_tools()
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
