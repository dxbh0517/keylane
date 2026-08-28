"""MCP client — persistent stdio sessions and tool registry integration."""

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


async def probe_mcp_server(srv: dict[str, Any]) -> dict[str, Any]:
    command = srv.get("command", "")
    args = list(srv.get("args", []))
    env = srv.get("env")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args, env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return {"tools": len(tools.tools), "command": command}


async def _get_session(sid: str, srv: dict[str, Any]) -> Any:
    global _stack
    if sid in _sessions:
        return _sessions[sid]

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if _stack is None:
        _stack = AsyncExitStack()
        await _stack.__aenter__()

    command = srv.get("command", "")
    args = list(srv.get("args", []))
    env = srv.get("env")
    params = StdioServerParameters(command=command, args=args, env=env)
    read, write = await _stack.enter_async_context(stdio_client(params))
    session = await _stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    _sessions[sid] = session
    return session


def _is_disabled(tool_name: str) -> bool:
    disabled = get_section("mcp").get("disabled_tools", [])
    return tool_name in disabled


async def _call_tool(sid: str, srv: dict[str, Any], tool_name: str, kwargs: dict[str, Any]) -> str:
    async with _lock:
        try:
            session = await _get_session(sid, srv)
            result = await session.call_tool(tool_name, kwargs)
            parts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(parts)
        except Exception:
            _sessions.pop(sid, None)
            session = await _get_session(sid, srv)
            result = await session.call_tool(tool_name, kwargs)
            parts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(parts)


async def load_mcp_tools(registry: ToolRegistry) -> int:
    cfg = mcp_settings()
    servers = cfg.get("servers", [])
    count = 0
    for srv in servers:
        sid = srv.get("id", "mcp")
        transport = srv.get("transport", "stdio")
        if transport != "stdio":
            logger.warning("MCP server %s: only stdio supported", sid)
            continue
        command = srv.get("command", "")
        if not command:
            logger.warning("MCP server %s: missing command", sid)
            continue
        try:
            info = await probe_mcp_server(srv)
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
            logger.info("MCP server %s: %s tools", sid, info.get("tools", 0))
        except Exception:  # noqa: BLE001
            logger.exception("failed to load MCP server %s", sid)
    return count


async def reload_mcp_tools(registry: ToolRegistry) -> int:
    global _stack, _sessions
    _sessions.clear()
    if _stack is not None:
        await _stack.aclose()
        _stack = None
    for name in list(registry._tools.keys()):
        if name.startswith("mcp."):
            del registry._tools[name]
    return await load_mcp_tools(registry)


async def shutdown_mcp() -> None:
    global _stack, _sessions
    _sessions.clear()
    if _stack is not None:
        await _stack.aclose()
        _stack = None
