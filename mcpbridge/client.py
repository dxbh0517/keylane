"""MCP client — persistent stdio and Streamable HTTP sessions.

This package is deliberately *not* named ``mcp``: that would shadow the
official ``mcp`` SDK on ``sys.path`` and every import below would fail.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from daemon.config import get_section, mcp_settings
from tools.registry import Tool, ToolRegistry

logger = logging.getLogger(__name__)

_sessions: dict[str, Any] = {}
_stack: AsyncExitStack | None = None
_lock = asyncio.Lock()


class McpError(RuntimeError):
    pass


def normalize_auth_header(value: str | None) -> str:
    """Ensure an ``Authorization`` value carries a scheme.

    Mailspring (Preferences → MCP Server) shows a bare UUID; sent without
    ``Bearer`` the server answers 401 and the whole server looks unreachable.
    """
    text = (value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith(("bearer ", "basic ")):
        return text
    if lowered.startswith("authorization:"):
        return text.split(":", 1)[1].strip() or text
    return f"Bearer {text}"


def normalize_headers(headers: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (headers or {}).items():
        if not value:
            continue
        if str(key).lower() == "authorization":
            out[str(key)] = normalize_auth_header(str(value))
        else:
            out[str(key)] = str(value)
    return out


def server_transport(srv: dict[str, Any]) -> str:
    """Normalize the transport name; a bare ``url`` implies HTTP."""
    transport = str(srv.get("transport", "")).strip().lower()
    if transport in {"http", "streamable-http", "streamable_http", "sse"}:
        return "http"
    if transport == "stdio":
        return "stdio"
    return "http" if srv.get("url") else "stdio"


def _server_headers(srv: dict[str, Any]) -> dict[str, str]:
    headers = dict(srv.get("headers") or {})
    token = srv.get("auth_header") or srv.get("token")
    if token and not any(k.lower() == "authorization" for k in headers):
        headers["Authorization"] = str(token)
    return normalize_headers(headers)


async def _open_session(sid: str, srv: dict[str, Any], stack: AsyncExitStack) -> Any:
    """Enter a ClientSession for *srv* on *stack* and initialize it."""
    from mcp import ClientSession

    if server_transport(srv) == "http":
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        url = str(srv.get("url", "")).strip()
        if not url:
            raise McpError(f"MCP server {sid}: http transport requires a url")
        http = create_mcp_http_client(headers=_server_headers(srv) or None)
        await stack.enter_async_context(http)
        streams = await stack.enter_async_context(streamable_http_client(url, http_client=http))
        read, write = streams[0], streams[1]
    else:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        command = str(srv.get("command", "")).strip()
        if not command:
            raise McpError(f"MCP server {sid}: stdio transport requires a command")
        params = StdioServerParameters(
            command=command,
            args=list(srv.get("args", [])),
            env=srv.get("env"),
        )
        read, write = await stack.enter_async_context(stdio_client(params))

    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def probe_mcp_server(srv: dict[str, Any]) -> dict[str, Any]:
    """Open a throwaway session just to count tools — used by health checks."""
    sid = str(srv.get("id", "mcp"))
    transport = server_transport(srv)
    async with AsyncExitStack() as stack:
        session = await _open_session(sid, srv, stack)
        tools = await session.list_tools()
        endpoint = srv.get("url") if transport == "http" else srv.get("command", "")
        return {"tools": len(tools.tools), "transport": transport, "endpoint": endpoint}


async def _get_session(sid: str, srv: dict[str, Any]) -> Any:
    global _stack
    if sid in _sessions:
        return _sessions[sid]
    if _stack is None:
        _stack = AsyncExitStack()
        await _stack.__aenter__()
    session = await _open_session(sid, srv, _stack)
    _sessions[sid] = session
    return session


def _is_disabled(tool_name: str) -> bool:
    disabled = get_section("mcp").get("disabled_tools", [])
    return tool_name in disabled


def _result_text(result: Any) -> str:
    parts = [c.text for c in result.content if hasattr(c, "text")]
    return "\n".join(parts)


async def _call_tool(sid: str, srv: dict[str, Any], tool_name: str, kwargs: dict[str, Any]) -> str:
    async with _lock:
        try:
            session = await _get_session(sid, srv)
            return _result_text(await session.call_tool(tool_name, kwargs))
        except Exception:  # noqa: BLE001 — one reconnect, then surface the error
            logger.info("MCP %s: session lost, reconnecting", sid)
            _sessions.pop(sid, None)
            session = await _get_session(sid, srv)
            return _result_text(await session.call_tool(tool_name, kwargs))


async def load_mcp_tools(registry: ToolRegistry) -> int:
    servers = mcp_settings().get("servers", [])
    count = 0
    for srv in servers:
        sid = srv.get("id", "mcp")
        try:
            async with _lock:
                session = await _get_session(sid, srv)
                tools = await session.list_tools()
            for tool in tools.tools:
                name = f"mcp.{sid}.{tool.name}"
                if _is_disabled(name):
                    continue

                async def _handler(
                    _sid: str = sid,
                    _srv: dict = srv,
                    _tool: str = tool.name,
                    **kwargs: Any,
                ) -> str:
                    return await _call_tool(_sid, _srv, _tool, kwargs)

                registry.register(
                    Tool(
                        name=name,
                        description=tool.description or f"MCP tool {tool.name}",
                        parameters=tool.inputSchema or {"type": "object", "properties": {}},
                        handler=_handler,
                    )
                )
                count += 1
            logger.info("MCP server %s (%s): %s tools", sid, server_transport(srv), len(tools.tools))
        except Exception:  # noqa: BLE001
            logger.exception("failed to load MCP server %s", sid)
    return count


async def reload_mcp_tools(registry: ToolRegistry) -> int:
    global _stack, _sessions
    _sessions.clear()
    if _stack is not None:
        await _stack.aclose()
        _stack = None
    for name in list(registry._tools.keys()):  # noqa: SLF001
        if name.startswith("mcp."):
            del registry._tools[name]  # noqa: SLF001
    return await load_mcp_tools(registry)


async def shutdown_mcp() -> None:
    global _stack, _sessions
    _sessions.clear()
    if _stack is not None:
        await _stack.aclose()
        _stack = None
