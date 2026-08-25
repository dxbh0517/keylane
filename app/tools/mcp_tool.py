"""Adapt tools exposed by an MCP plugin into assistant tools.

Installing an MCP server is therefore enough to teach the assistant new tricks:
its tool list is discovered, namespaced under the plugin id, and called through
the same policy gate as everything else.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.plugins.base import BasePlugin
from app.plugins.mcp_client import mcp_call_tool, mcp_call_tool_http, mcp_list_tools, mcp_list_tools_http
from app.tools.base import BaseTool, ToolDanger, ToolResult

logger = logging.getLogger(__name__)

# MCP tool names that read state; everything else is treated as sensitive.
READ_ONLY_HINTS = (
    "list",
    "get",
    "search",
    "info",
    "status",
    "read",
    "describe",
    "which",
    "validate",
    "fetch",
    "stats",
)


def _guess_danger(name: str) -> ToolDanger:
    lowered = name.lower()
    if any(lowered.startswith(hint) or f"_{hint}" in lowered for hint in READ_ONLY_HINTS):
        return ToolDanger.SAFE
    return ToolDanger.SENSITIVE


class McpTool(BaseTool):
    """One tool from one MCP server."""

    def __init__(
        self,
        *,
        plugin_id: str,
        plugin: BasePlugin,
        tool_name: str,
        description: str,
        schema: dict[str, Any] | None,
    ) -> None:
        self.name = f"{plugin_id}.{tool_name}"
        self.description = (description or f"{tool_name} on the {plugin_id} MCP server.")[:600]
        self.danger = _guess_danger(tool_name)
        self.category = f"mcp:{plugin_id}"
        self.source = plugin_id
        self._plugin = plugin
        self._tool_name = tool_name
        self._schema = schema or {"type": "object", "properties": {}}

    def parameters(self) -> dict[str, Any]:
        return self._schema

    async def run(self, args: dict[str, Any]) -> ToolResult:
        descriptor = self._plugin.mcp_descriptor() or {}
        transport = str(descriptor.get("transport") or "stdio")
        try:
            if transport == "http":
                url = str(descriptor.get("url") or "").strip()
                if not url:
                    return ToolResult.failure(f"{self.source} has no MCP URL configured.")
                auth = str(descriptor.get("auth_header") or "").strip()
                headers = {"Authorization": auth} if auth else {}
                payload = await mcp_call_tool_http(
                    url,
                    self._tool_name,
                    dict(args or {}),
                    headers=headers,
                )
            else:
                command = descriptor.get("command")
                if not command:
                    return ToolResult.failure(f"{self.source} has no MCP command configured.")
                try:
                    command = self._plugin._command()  # resolves PATH / ~/.local/bin
                except Exception:  # noqa: BLE001
                    pass
                payload = await mcp_call_tool(
                    str(command),
                    self._tool_name,
                    dict(args or {}),
                    args=descriptor.get("args") or [],
                    env=getattr(self._plugin, "_env", lambda: None)(),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP tool %s failed: %s", self.name, exc)
            return ToolResult.failure(f"{self.name} failed: {exc}")

        if isinstance(payload, str):
            text = payload
            data: dict[str, Any] = {}
        elif isinstance(payload, dict):
            structured = payload.get("structured")
            content = payload.get("content")
            text = (
                "\n".join(str(c) for c in content if isinstance(c, str))
                if isinstance(content, list)
                else json.dumps(payload, default=str)
            )
            data = structured if isinstance(structured, dict) else {"result": payload}
        else:
            text = json.dumps(payload, default=str)
            data = {"result": payload}

        return ToolResult.success(text[:12000], data=data)


async def mcp_tools_for_plugin(plugin_id: str, plugin: BasePlugin) -> list[McpTool]:
    descriptor = plugin.mcp_descriptor() or {}
    transport = str(descriptor.get("transport") or "stdio")
    try:
        if transport == "http":
            url = str(descriptor.get("url") or "").strip()
            if not url:
                return []
            auth = str(descriptor.get("auth_header") or "").strip()
            headers = {"Authorization": auth} if auth else {}
            entries = await mcp_list_tools_http(url, headers=headers)
        else:
            command = descriptor.get("command")
            if not command:
                return []
            try:
                command = plugin._command()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            entries = await mcp_list_tools(
                str(command),
                descriptor.get("args") or [],
                env=getattr(plugin, "_env", lambda: None)(),
            )
    except Exception as exc:  # noqa: BLE001
        logger.info("Could not list MCP tools for %s: %s", plugin_id, exc)
        return []
    tools: list[McpTool] = []
    for entry in entries:
        name = entry.get("name")
        if not name:
            continue
        tools.append(
            McpTool(
                plugin_id=plugin_id,
                plugin=plugin,
                tool_name=str(name),
                description=str(entry.get("description") or ""),
                schema=entry.get("input_schema"),
            )
        )
    return tools
