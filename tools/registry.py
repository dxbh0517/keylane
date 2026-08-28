"""Tool registry — self-registering builtin tools."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., Any] | Callable[..., Awaitable[Any]]

# Common model mistakes -> schema field names
_ARG_ALIASES: dict[str, str] = {
    "query": "question",
    "q": "question",
    "text": "question",
    "prompt": "question",
    "search": "question",
}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    dangerous: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def describe_for_prompt(self) -> str:
        """Compact tool list for NPU context limits (~1024 tokens)."""
        lines = []
        for t in self._tools.values():
            props = t.parameters.get("properties", {})
            keys = ", ".join(props.keys()) if props else ""
            lines.append(f"- {t.name}({keys}): {t.description}")
        return "\n".join(lines)

    def _normalize_arguments(self, tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
        props = set(tool.parameters.get("properties", {}).keys())
        out = dict(arguments)
        for alias, canonical in _ARG_ALIASES.items():
            if alias in out and canonical in props and canonical not in out:
                out[canonical] = out.pop(alias)
        return out

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"unknown tool: {name}"})
        args = self._normalize_arguments(tool, arguments)

        from daemon.config import permission_mode

        mode = permission_mode(name)
        if mode == "deny" and (tool.dangerous or name in {"shell", "memory_write", "schedule_task"}):
            return json.dumps({"error": f"permission denied for {name}"})

        if mode == "ask" and (tool.dangerous or name in {"shell", "memory_write", "schedule_task"}):
            import asyncio
            from daemon.permissions import create_pending, wait_pending

            pending = create_pending(name, args)
            if pending and pending.id:
                if on_event:
                    on_event(
                        "permission",
                        {
                            "id": pending.id,
                            "tool": name,
                            "arguments": args,
                            "message": f"Allow {name}?",
                        },
                    )
                ok = await asyncio.to_thread(wait_pending, pending)
                if not ok:
                    return json.dumps({"error": f"permission denied for {name}"})

        try:
            result = tool.handler(**args)
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except TypeError as exc:
            # Help the model recover on the next loop iteration.
            sig = inspect.signature(tool.handler)
            expected = [
                p.name
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            ]
            return json.dumps(
                {
                    "error": str(exc),
                    "expected_arguments": expected,
                    "received": list(args.keys()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("tool %s failed", name)
            return json.dumps({"error": str(exc)})


_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _registry
