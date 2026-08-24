"""Tool contract for the Keylane assistant.

A *tool* is a single, well-described capability the NPU assistant may invoke
directly: opening an application, searching the web, reading a file, handing a
job to Claude Code, and so on.  Tools are deliberately narrow and typed — the
model emits a name plus JSON arguments, and Python validates and executes.

Danger levels drive the confirmation policy:

``SAFE``       read-only or trivially reversible — runs without asking.
``SENSITIVE``  writes/sends something (files, email, notifications).
``DANGEROUS``  arbitrary execution or destructive potential — always gated.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


class ToolDanger(str, Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    DANGEROUS = "dangerous"


DANGER_ORDER = {ToolDanger.SAFE: 0, ToolDanger.SENSITIVE: 1, ToolDanger.DANGEROUS: 2}


class ToolSpec(BaseModel):
    """Everything the model and the control panel need to know about a tool."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    danger: ToolDanger = ToolDanger.SAFE
    category: str = "general"
    source: str = "builtin"
    enabled: bool = True
    requires_confirmation: bool = False
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    def prompt_line(self, description_chars: int = 110) -> str:
        """One compact line per tool for the assistant prompt.

        MCP servers often ship multi-paragraph descriptions with ``Args:`` and
        ``Returns:`` sections. Pasted verbatim, forty of those bury the actual
        instructions — so each line is trimmed to its first sentence.
        """
        props = (self.parameters or {}).get("properties") or {}
        required = set((self.parameters or {}).get("required") or [])
        args = ", ".join(
            f"{key}{'' if key in required else '?'}:{(val or {}).get('type', 'string')}"
            for key, val in list(props.items())[:6]
        )
        if len(props) > 6:
            args += ", …"

        summary = " ".join((self.description or "").split())
        # First sentence, or a hard trim if there is no sentence break.
        cut = summary.find(". ")
        if 0 < cut < description_chars:
            summary = summary[: cut + 1]
        elif len(summary) > description_chars:
            summary = summary[: description_chars - 1].rstrip(" ,;:") + "…"
        return f"- {self.name}({args}) — {summary}"


class ToolResult(BaseModel):
    ok: bool
    output: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)

    @classmethod
    def success(
        cls,
        output: str,
        data: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
    ) -> "ToolResult":
        return cls(ok=True, output=output, data=data or {}, artifacts=artifacts or [])

    @classmethod
    def failure(cls, error: str, data: dict[str, Any] | None = None) -> "ToolResult":
        return cls(ok=False, output="", error=error, data=data or {})


class ToolError(RuntimeError):
    """Raised by a tool when its arguments or environment are unusable."""


class BaseTool(ABC):
    """Base class for a single assistant capability."""

    name: str = ""
    description: str = ""
    danger: ToolDanger = ToolDanger.SAFE
    category: str = "general"
    source: str = "builtin"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def availability(self) -> str | None:
        """Return ``None`` when usable, else a human reason why it is not."""
        return None

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters(),
            danger=self.danger,
            category=self.category,
            source=self.source,
            unavailable_reason=self.availability(),
        )

    @abstractmethod
    async def run(self, args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


class FunctionTool(BaseTool):
    """Adapter that turns a plain async callable into a tool."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[ToolResult] | ToolResult],
        danger: ToolDanger = ToolDanger.SAFE,
        category: str = "general",
        source: str = "builtin",
        availability: Callable[[], str | None] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.danger = danger
        self.category = category
        self.source = source
        self._parameters = parameters
        self._handler = handler
        self._availability = availability

    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def availability(self) -> str | None:
        return self._availability() if self._availability else None

    async def run(self, args: dict[str, Any]) -> ToolResult:
        result = self._handler(args)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, str):
            return ToolResult.success(result)
        return ToolResult.success("", data=dict(result or {}))


def object_schema(
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def string_prop(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "string", "description": description, **extra}


def int_prop(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "integer", "description": description, **extra}


def bool_prop(description: str, **extra: Any) -> dict[str, Any]:
    return {"type": "boolean", "description": description, **extra}
