"""Generic MCP plugin — drive any stdio MCP server as a gateway worker/utility."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from app.plugins.base import (
    BasePlugin,
    PluginHealth,
    PluginKind,
    SettingField,
    SettingType,
)
from app.plugins.mcp_client import McpError, mcp_call_tool, mcp_list_tools
from app.schemas import RouteDecision, WorkerEvidence, WorkerResult

logger = logging.getLogger(__name__)


class McpPlugin(BasePlugin):
    """
    MCP-backed plugin.

    Manifest keys (settings / plugin.toml):
      command          executable (e.g. comfy-mcp)
      args             JSON list of CLI args
      health_tool      tool name for health (default server_info)
      run_tool         tool name for worker execution (default generate_image)
      env              JSON object of extra env vars
      worker_id        route worker name (defaults to plugin id)
      cloud            whether this MCP talks to cloud services
    """

    kind = PluginKind.MCP
    removable = True

    def __init__(
        self,
        *,
        plugin_id: str,
        name: str,
        description: str = "",
        settings: dict[str, Any] | None = None,
        author: str = "community",
        version: str = "0.1.0",
        homepage: str | None = None,
        worker_id: str | None = None,
        cloud: bool = False,
    ) -> None:
        self.id = plugin_id
        self.name = name
        self.description = description
        self.author = author
        self.version = version
        self.homepage = homepage
        self.worker_id = worker_id or plugin_id
        self.cloud = cloud
        super().__init__(settings)

    def default_settings(self) -> dict[str, Any]:
        return {
            "command": "comfy-mcp",
            "args": "[]",
            "health_tool": "server_info",
            "run_tool": "generate_image",
            "env": "{}",
            **super().default_settings(),
        }

    def settings_schema(self) -> list[SettingField]:
        return [
            SettingField(
                key="command",
                label="MCP command",
                type=SettingType.STRING,
                description="Executable that speaks MCP over stdio",
                default="comfy-mcp",
                required=True,
            ),
            SettingField(
                key="args",
                label="Command args (JSON array)",
                type=SettingType.JSON,
                default="[]",
            ),
            SettingField(
                key="health_tool",
                label="Health tool",
                type=SettingType.STRING,
                default="server_info",
            ),
            SettingField(
                key="run_tool",
                label="Run tool",
                type=SettingType.STRING,
                description="Tool invoked for worker tasks",
                default="generate_image",
            ),
            SettingField(
                key="env",
                label="Extra environment (JSON object)",
                type=SettingType.JSON,
                default="{}",
            ),
        ]

    def mcp_descriptor(self) -> dict[str, Any]:
        return {
            "transport": "stdio",
            "command": self.settings.get("command"),
            "args": self._parse_json(self.settings.get("args"), []),
            "health_tool": self.settings.get("health_tool", "server_info"),
            "run_tool": self.settings.get("run_tool", "generate_image"),
        }

    @staticmethod
    def _parse_json(value: Any, fallback: Any) -> Any:
        if isinstance(value, (list, dict)):
            return value
        if not value:
            return fallback
        try:
            return json.loads(str(value))
        except json.JSONDecodeError:
            return fallback

    def _command(self) -> str:
        cmd = str(self.settings.get("command") or "").strip()
        if not cmd:
            raise McpError("MCP command is not configured")
        resolved = shutil.which(cmd) or (cmd if Path(cmd).exists() else None)
        if not resolved:
            raise McpError(f"MCP command not found on PATH: {cmd}")
        return resolved

    def _args(self) -> list[str]:
        raw = self._parse_json(self.settings.get("args"), [])
        return [str(x) for x in raw] if isinstance(raw, list) else []

    def _env(self) -> dict[str, str] | None:
        raw = self._parse_json(self.settings.get("env"), {})
        if not isinstance(raw, dict) or not raw:
            return None
        return {str(k): str(v) for k, v in raw.items()}

    async def health(self) -> PluginHealth:
        try:
            cmd = self._command()
            tool = str(self.settings.get("health_tool") or "server_info")
            # Prefer a lightweight tool call; fall back to list_tools.
            try:
                payload = await mcp_call_tool(
                    cmd, tool, {}, args=self._args(), env=self._env()
                )
                return PluginHealth(
                    ok=True,
                    detail=f"MCP healthy via {tool}",
                    metadata={"result": _short(payload)},
                )
            except Exception as tool_exc:  # noqa: BLE001
                tools = await mcp_list_tools(cmd, self._args())
                names = [t.get("name") for t in tools]
                return PluginHealth(
                    ok=True,
                    detail=f"MCP reachable ({len(names)} tools); health tool failed: {tool_exc}",
                    metadata={"tools": names},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP plugin %s health failed: %s", self.id, exc)
            return PluginHealth(ok=False, detail=str(exc))

    async def run(self, decision: RouteDecision) -> WorkerResult:
        try:
            cmd = self._command()
            tool = str(self.settings.get("run_tool") or "generate_image")
            arguments = self._build_arguments(decision, tool)
            payload = await mcp_call_tool(
                cmd, tool, arguments, args=self._args(), env=self._env()
            )
            summary, output_path, dims = _extract_media(payload, decision)
            evidence = WorkerEvidence(
                worker=self.worker_id or self.id,
                action=decision.action,
                response=summary if isinstance(summary, str) else json.dumps(summary)[:8000],
                stdout=summary if isinstance(summary, str) else json.dumps(payload)[:8000],
                exit_code=0,
                output_path=output_path,
                output_dimensions=dims,
                metadata={"mcp_tool": tool, "mcp_command": cmd},
            )
            return WorkerResult(
                success=True,
                evidence=evidence,
                summary=summary if isinstance(summary, str) else str(summary)[:2000],
                raw=payload,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP plugin %s run failed", self.id)
            evidence = WorkerEvidence(
                worker=self.worker_id or self.id,
                action=decision.action,
                stderr=str(exc),
                exit_code=1,
            )
            return WorkerResult(success=False, evidence=evidence, summary=str(exc))

    def _build_arguments(self, decision: RouteDecision, tool: str) -> dict[str, Any]:
        args = dict(decision.arguments or {})
        # Common Comfy MCP generate_image shape
        if tool in {"generate_image", "run_template"}:
            if "prompt" not in args:
                args["prompt"] = decision.instruction
            if decision.workflow and "workflow" not in args:
                args["workflow"] = decision.workflow
        elif tool == "run_workflow":
            if "prompt" not in args and "workflow" not in args:
                args["prompt"] = decision.instruction
        else:
            args.setdefault("instruction", decision.instruction)
            args.setdefault("prompt", decision.instruction)
        return args


def _short(value: Any, limit: int = 500) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


def _extract_media(
    payload: Any, decision: RouteDecision
) -> tuple[str, str | None, dict[str, int] | None]:
    output_path = None
    dims = None
    if isinstance(payload, dict):
        structured = payload.get("structured") if "structured" in payload else payload
        if isinstance(structured, dict):
            for key in ("output_path", "path", "file", "filename", "image_path"):
                if structured.get(key):
                    output_path = str(structured[key])
                    break
            outputs = structured.get("outputs") or structured.get("files")
            if not output_path and isinstance(outputs, list) and outputs:
                first = outputs[0]
                if isinstance(first, str):
                    output_path = first
                elif isinstance(first, dict):
                    output_path = str(
                        first.get("path") or first.get("filename") or first.get("file") or ""
                    ) or None
            w = structured.get("width") or decision.arguments.get("width")
            h = structured.get("height") or decision.arguments.get("height")
            if w and h:
                dims = {"width": int(w), "height": int(h)}
        content = payload.get("content")
        if isinstance(content, list) and content:
            summary = " ".join(str(c) for c in content if isinstance(c, str))[:2000]
        else:
            summary = json.dumps(payload, default=str)[:2000]
    else:
        summary = str(payload)[:2000]
        # Heuristic: look for a filesystem path in text
        for token in summary.split():
            if token.startswith("/") and ("." in Path(token).name):
                candidate = Path(token.strip(".,)"))
                if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                    output_path = str(candidate)
                    break
    if not summary:
        summary = f"MCP tool completed ({output_path or 'no path'})"
    return summary, output_path, dims


def mcp_plugin_from_manifest(data: dict[str, Any], settings: dict[str, Any] | None = None) -> McpPlugin:
    merged = {**(data.get("settings") or {}), **(settings or {})}
    if data.get("command"):
        merged.setdefault("command", data["command"])
    if data.get("args") is not None:
        merged.setdefault(
            "args",
            data["args"] if isinstance(data["args"], str) else json.dumps(data["args"]),
        )
    for key in ("health_tool", "run_tool", "env"):
        if key in data:
            val = data[key]
            merged.setdefault(key, val if isinstance(val, str) else json.dumps(val))
    return McpPlugin(
        plugin_id=data["id"],
        name=data.get("name") or data["id"],
        description=data.get("description") or "",
        settings=merged,
        author=data.get("author") or "community",
        version=data.get("version") or "0.1.0",
        homepage=data.get("homepage"),
        worker_id=data.get("worker_id") or data["id"],
        cloud=bool(data.get("cloud", False)),
    )
