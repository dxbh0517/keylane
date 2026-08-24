"""Tool registry — collects built-in tools plus everything plugins contribute.

Any plugin may expose tools by implementing ``tools()``; MCP plugins get their
server's tools adapted automatically, so installing an MCP server is enough to
give the assistant new abilities. Policy (allow/deny, danger gating) is applied
here, in Python — never in the prompt.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.assistant_settings import load_assistant_settings
from app.config import AppConfig, get_config
from app.plugins.base import BasePlugin, PluginKind
from app.plugins.registry import PluginRegistry, get_plugin_registry
from app.tools.base import DANGER_ORDER, BaseTool, ToolDanger, ToolResult, ToolSpec
from app.tools.builtin.delegation import delegation_tools
from app.tools.builtin.desktop import desktop_tools
from app.tools.builtin.email import email_tools
from app.tools.builtin.files import file_tools
from app.tools.builtin.shell import shell_tools
from app.tools.builtin.web import web_tools

logger = logging.getLogger(__name__)


class ConfirmationRequired(Exception):
    """Raised when a tool call needs explicit user approval before it runs."""

    def __init__(self, tool: str, args: dict[str, Any], danger: ToolDanger) -> None:
        super().__init__(f"Tool '{tool}' requires confirmation")
        self.tool = tool
        self.args = args
        self.danger = danger


class ToolRegistry:
    def __init__(
        self,
        config: AppConfig | None = None,
        plugins: PluginRegistry | None = None,
    ) -> None:
        self.config = config or get_config()
        self.plugins = plugins or get_plugin_registry(self.config)
        self._builtin: dict[str, BaseTool] = {}
        self._plugin_tools: dict[str, BaseTool] = {}
        self._mcp_loaded = False
        self._load_builtins()
        self._load_plugin_tools()

    # ---------------------------------------------------------------- loading

    def _load_builtins(self) -> None:
        tools: list[BaseTool] = []
        tools += desktop_tools()
        tools += web_tools()
        tools += file_tools()
        tools += shell_tools()
        tools += email_tools()
        tools += delegation_tools(self.plugins, self.config)
        self._builtin = {tool.name: tool for tool in tools}

    def _load_plugin_tools(self) -> None:
        """Pull in tools declared synchronously by enabled plugins."""
        self._plugin_tools = {}
        for plugin_id, plugin in self.plugins.items():
            if not self.plugins.is_enabled(plugin_id):
                continue
            getter = getattr(plugin, "tools", None)
            if getter is None:
                continue
            try:
                contributed = getter() or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("Plugin %s failed to list tools: %s", plugin_id, exc)
                continue
            for tool in contributed:
                if not isinstance(tool, BaseTool):
                    continue
                tool.source = plugin_id
                name = tool.name
                if name in self._builtin:
                    name = f"{plugin_id}_{name}"
                    tool.name = name
                self._plugin_tools[name] = tool

    async def load_mcp_tools(self, *, force: bool = False) -> int:
        """Discover tools exposed by enabled MCP plugins (one round-trip each)."""
        if self._mcp_loaded and not force:
            return 0
        from app.tools.mcp_tool import mcp_tools_for_plugin

        self._mcp_loaded = True
        discovered = 0
        for plugin_id, plugin in self.plugins.items():
            if not self.plugins.is_enabled(plugin_id):
                continue
            # PluginKind is a str enum, so this compares cleanly.
            if plugin.kind != PluginKind.MCP:
                continue
            try:
                tools = await mcp_tools_for_plugin(plugin_id, plugin)
            except Exception as exc:  # noqa: BLE001
                logger.info("Could not list MCP tools for %s: %s", plugin_id, exc)
                continue
            for tool in tools:
                self._plugin_tools[tool.name] = tool
                discovered += 1
        return discovered

    def reload(self) -> None:
        self._mcp_loaded = False
        self._load_builtins()
        self._load_plugin_tools()

    # ----------------------------------------------------------------- lookup

    def _all(self) -> dict[str, BaseTool]:
        return {**self._builtin, **self._plugin_tools}

    def get(self, name: str) -> BaseTool | None:
        return self._all().get(name)

    def all_names(self) -> set[str]:
        """Every registered tool name, for interpreting model output."""
        return set(self._all())

    def policy_blocks(self, name: str) -> str | None:
        policy = load_assistant_settings().tools
        if not policy.enabled:
            return "The assistant tool layer is switched off."
        if name in policy.deny:
            return f"Tool '{name}' is on the deny list."
        if policy.allow and name not in policy.allow:
            return f"Tool '{name}' is not on the allow list."
        return None

    def needs_confirmation(self, tool: BaseTool) -> bool:
        policy = load_assistant_settings().tools
        if tool.name in policy.auto_confirm:
            return False
        try:
            threshold = ToolDanger(policy.confirm_danger_at)
        except ValueError:
            threshold = ToolDanger.SENSITIVE
        return DANGER_ORDER[tool.danger] >= DANGER_ORDER[threshold]

    def specs(self, *, include_unavailable: bool = True) -> list[ToolSpec]:
        out: list[ToolSpec] = []
        for name, tool in sorted(self._all().items()):
            spec = tool.spec()
            spec.name = name
            blocked = self.policy_blocks(name)
            spec.enabled = blocked is None
            if blocked and not spec.unavailable_reason:
                spec.unavailable_reason = blocked
            spec.requires_confirmation = self.needs_confirmation(tool)
            if not include_unavailable and (not spec.enabled or not spec.available):
                continue
            out.append(spec)
        return out

    def usable_specs(self) -> list[ToolSpec]:
        """Tools the assistant may actually be told about."""
        return [s for s in self.specs() if s.enabled and s.available]

    # A 1.5B model given an 8,000-token catalogue stops following instructions
    # altogether. This bounds the *plugin and MCP* tail — built-in tools are
    # always listed, because losing delegate_to_worker would be worse than a
    # slightly longer prompt.
    PROMPT_BUDGET_CHARS = 1800

    def prompt_catalog(self, limit: int = 60, budget: int | None = None) -> str:
        """The tool list as the model sees it, kept small enough to follow.

        Built-in tools are listed before plugin and MCP tools: a server that
        exposes forty tools must not crowd out ``delegate_to_worker``.
        """
        budget = self.PROMPT_BUDGET_CHARS if budget is None else budget
        specs = self.usable_specs()
        builtin = {name for name in self._builtin}
        specs.sort(key=lambda s: (s.name not in builtin, s.category, s.name))
        specs = specs[:limit]

        by_category: dict[str, list[str]] = {}
        used = 0
        dropped = 0
        for spec in specs:
            line = spec.prompt_line()
            if used + len(line) > budget and spec.name not in builtin:
                dropped += 1
                continue
            by_category.setdefault(spec.category, []).append(line)
            used += len(line) + 1

        if not by_category:
            return "(no tools are currently available)"

        blocks = [
            f"{category}:\n" + "\n".join(lines)
            for category, lines in sorted(by_category.items())
        ]
        if dropped:
            blocks.append(
                f"({dropped} more plugin tools exist. Ask for them by name if you "
                f"need one, or delegate the job to the worker that owns them.)"
            )
        return "\n\n".join(blocks)

    # ---------------------------------------------------------------- calling

    async def call(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        confirmed: bool = False,
    ) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            close = [n for n in self._all() if name in n or n in name]
            hint = f" Did you mean: {', '.join(close[:3])}?" if close else ""
            return ToolResult.failure(f"No tool named '{name}'.{hint}")

        blocked = self.policy_blocks(name)
        if blocked:
            return ToolResult.failure(blocked)

        unavailable = tool.availability()
        if unavailable:
            return ToolResult.failure(f"'{name}' is unavailable: {unavailable}")

        if self.needs_confirmation(tool) and not confirmed:
            raise ConfirmationRequired(name, args or {}, tool.danger)

        try:
            return await tool.run(dict(args or {}))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %s raised", name)
            return ToolResult.failure(f"{name} raised an error: {exc}")


_registry: ToolRegistry | None = None


def get_tool_registry(config: AppConfig | None = None) -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry(config)
    return _registry


def reload_tool_registry(config: AppConfig | None = None) -> ToolRegistry:
    global _registry
    _registry = ToolRegistry(config)
    return _registry
