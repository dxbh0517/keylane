"""Plugin contracts for native workers and MCP-backed integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas import RouteDecision, WorkerResult


class PluginKind(str, Enum):
    NATIVE = "native"
    MCP = "mcp"
    UTILITY = "utility"


class SettingType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    PATH = "path"
    SELECT = "select"
    SECRET = "secret"
    JSON = "json"


class SettingField(BaseModel):
    key: str
    label: str
    type: SettingType = SettingType.STRING
    description: str = ""
    default: Any = None
    required: bool = False
    options: list[str] = Field(default_factory=list)
    placeholder: str = ""


class PluginHealth(BaseModel):
    ok: bool
    detail: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PluginInfo(BaseModel):
    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    kind: PluginKind
    author: str = "built-in"
    enabled: bool = True
    worker_id: str | None = None
    cloud: bool = False
    configurable: bool = True
    removable: bool = False
    settings_schema: list[SettingField] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    health: PluginHealth | None = None
    mcp: dict[str, Any] | None = None
    homepage: str | None = None


class BasePlugin(ABC):
    """Base class for gateway plugins."""

    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    kind: PluginKind = PluginKind.NATIVE
    author: str = "built-in"
    worker_id: str | None = None
    cloud: bool = False
    removable: bool = False
    homepage: str | None = None

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.settings: dict[str, Any] = {**self.default_settings(), **(settings or {})}

    def default_settings(self) -> dict[str, Any]:
        return {field.key: field.default for field in self.settings_schema()}

    def settings_schema(self) -> list[SettingField]:
        return []

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        schema_keys = {f.key for f in self.settings_schema()}
        for key, value in data.items():
            if schema_keys and key not in schema_keys:
                continue
            self.settings[key] = value
        return self.settings

    def info(self, *, enabled: bool = True, health: PluginHealth | None = None) -> PluginInfo:
        return PluginInfo(
            id=self.id,
            name=self.name,
            version=self.version,
            description=self.description,
            kind=self.kind,
            author=self.author,
            enabled=enabled,
            worker_id=self.worker_id,
            cloud=self.cloud,
            configurable=True,
            removable=self.removable,
            settings_schema=self.settings_schema(),
            settings=self.settings,
            health=health,
            mcp=self.mcp_descriptor() if self.kind == PluginKind.MCP else None,
            homepage=self.homepage,
        )

    def mcp_descriptor(self) -> dict[str, Any] | None:
        return None

    @abstractmethod
    async def health(self) -> PluginHealth:
        raise NotImplementedError

    async def run(self, decision: RouteDecision) -> WorkerResult:
        raise NotImplementedError(
            f"Plugin '{self.id}' does not implement run(); it is not a worker."
        )

    async def close(self) -> None:
        return None
