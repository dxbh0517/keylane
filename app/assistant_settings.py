"""Assistant settings — tool policy, delegation, search and email credentials.

Stored in ``config/assistant.toml`` so the control panel and hand edits agree.
Secrets (SMTP password) are read from the environment when the value looks like
``env:VAR_NAME`` so the file itself stays safe to share.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.config import ROOT

SETTINGS_PATH = ROOT / "config" / "assistant.toml"

# Commands the shell tool may run without an explicit allowlist entry.
DEFAULT_SHELL_ALLOWLIST = [
    "ls",
    "cat",
    "head",
    "tail",
    "grep",
    "rg",
    "find",
    "wc",
    "df",
    "du",
    "free",
    "uptime",
    "date",
    "whoami",
    "hostname",
    "uname",
    "ps",
    "systemctl",
    "journalctl",
    "git",
    "python3",
    "pip",
    "nmcli",
    "ip",
    "lsblk",
    "sensors",
    "playerctl",
    "wpctl",
    "pactl",
    "notify-send",
    "xdg-open",
    "gio",
    "flatpak",
]


class ToolPolicy(BaseModel):
    enabled: bool = True
    """Master switch for the whole assistant tool layer."""

    allow: list[str] = Field(default_factory=list)
    """When non-empty, only these tool names may run."""

    deny: list[str] = Field(default_factory=list)
    """Tool names that may never run, regardless of ``allow``."""

    confirm_danger_at: str = "sensitive"
    """Lowest danger level that needs user confirmation: safe|sensitive|dangerous."""

    auto_confirm: list[str] = Field(default_factory=list)
    """Tool names that skip confirmation even above ``confirm_danger_at``."""

    max_steps: int = 6
    """Maximum tool calls the assistant may chain for one request."""


class ShellPolicy(BaseModel):
    enabled: bool = True
    allowlist: list[str] = Field(default_factory=lambda: list(DEFAULT_SHELL_ALLOWLIST))
    timeout_seconds: int = 60
    working_directory: str = "~"


class SearchSettings(BaseModel):
    engine: str = "duckduckgo"
    """duckduckgo | searxng | none"""

    searxng_url: str = "http://127.0.0.1:8888"
    max_results: int = 5
    timeout_seconds: int = 15


class EmailSettings(BaseModel):
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = ""
    """Literal value, or ``env:VAR_NAME`` to read from the environment."""

    from_address: str = ""
    from_name: str = ""
    allowed_recipients: list[str] = Field(default_factory=list)
    """When non-empty, mail may only be sent to these addresses or domains."""

    timeout_seconds: int = 30

    def resolved_password(self) -> str:
        value = (self.password or "").strip()
        if value.startswith("env:"):
            return os.environ.get(value[4:], "")
        return value


class SpeechSettings(BaseModel):
    enabled: bool = False
    """Show the read-aloud button and allow the speak tool."""

    engine: str = ""
    """piper | espeak | flite. Empty picks the best one installed."""

    voice: str = ""
    """Engine-specific voice id. Empty picks the engine's first voice."""

    rate: int = 100
    """Speaking rate as a percentage of the engine's normal speed."""

    pitch: int = 50
    """0-99, eSpeak only."""

    auto_speak: bool = False
    """Read every answer aloud as soon as it arrives."""


class DelegationSettings(BaseModel):
    enabled: bool = True
    """Let the assistant hand work to configured AI tools (Claude Code, …)."""

    follow_up: bool = True
    """Re-check a delegated result against the original request."""

    max_delegations: int = 2
    prefer: list[str] = Field(default_factory=list)
    """Ordered worker preference when several could do the job."""


class AssistantSettings(BaseModel):
    persona: str = ""
    """Extra sentences appended to the assistant system prompt."""

    tools: ToolPolicy = Field(default_factory=ToolPolicy)
    shell: ShellPolicy = Field(default_factory=ShellPolicy)
    search: SearchSettings = Field(default_factory=SearchSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    speech: SpeechSettings = Field(default_factory=SpeechSettings)
    delegation: DelegationSettings = Field(default_factory=DelegationSettings)

    def sanitized(self) -> dict[str, Any]:
        """Model dump with the SMTP password masked, for the API/control panel."""
        data = self.model_dump()
        password = (self.email.password or "").strip()
        if password.startswith("env:"):
            data["email"]["password"] = password
        elif password:
            data["email"]["password"] = "********"
        return data


_cache: AssistantSettings | None = None


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_assistant_settings(*, refresh: bool = False) -> AssistantSettings:
    global _cache
    if _cache is not None and not refresh:
        return _cache
    raw = _read_toml(SETTINGS_PATH)
    _cache = AssistantSettings(
        persona=str(raw.get("persona") or ""),
        tools=ToolPolicy(**(raw.get("tools") or {})),
        shell=ShellPolicy(**(raw.get("shell") or {})),
        search=SearchSettings(**(raw.get("search") or {})),
        email=EmailSettings(**(raw.get("email") or {})),
        speech=SpeechSettings(**(raw.get("speech") or {})),
        delegation=DelegationSettings(**(raw.get("delegation") or {})),
    )
    return _cache


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        inner = ", ".join(_toml_value(v) for v in value)
        return f"[{inner}]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def save_assistant_settings(settings: AssistantSettings) -> AssistantSettings:
    global _cache
    lines = [
        "# Keylane assistant configuration.",
        "# Tool policy, delegation, web search and outgoing mail.",
        "# Secrets may be written as \"env:VAR_NAME\" to read from the environment.",
        "",
        f"persona = {_toml_value(settings.persona)}",
        "",
    ]
    for section, model in (
        ("tools", settings.tools),
        ("shell", settings.shell),
        ("search", settings.search),
        ("email", settings.email),
        ("speech", settings.speech),
        ("delegation", settings.delegation),
    ):
        lines.append(f"[{section}]")
        for key, value in model.model_dump().items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")

    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text("\n".join(lines), encoding="utf-8")
    _cache = settings
    return settings


class AssistantSettingsUpdate(BaseModel):
    persona: str | None = None
    tools: dict[str, Any] | None = None
    shell: dict[str, Any] | None = None
    search: dict[str, Any] | None = None
    email: dict[str, Any] | None = None
    speech: dict[str, Any] | None = None
    delegation: dict[str, Any] | None = None


def update_assistant_settings(update: AssistantSettingsUpdate) -> AssistantSettings:
    current = load_assistant_settings()
    data = current.model_dump()
    payload = update.model_dump(exclude_none=True)

    if "persona" in payload:
        data["persona"] = payload["persona"]
    for section in ("tools", "shell", "search", "speech", "delegation"):
        if section in payload:
            data[section] = {**data[section], **payload[section]}
    if "email" in payload:
        incoming = dict(payload["email"])
        # A masked password means "leave the stored one alone".
        if incoming.get("password") in {"********", None}:
            incoming.pop("password", None)
        data["email"] = {**data["email"], **incoming}

    return save_assistant_settings(AssistantSettings(**data))


def set_tool_policy(
    name: str,
    *,
    enabled: bool | None = None,
    auto_confirm: bool | None = None,
) -> AssistantSettings:
    """Flip one tool's allow/deny and confirmation exemption.

    ``enabled`` is expressed through the deny list rather than the allow list,
    so switching a single tool off does not implicitly disable every other one.
    """
    settings = load_assistant_settings()
    policy = settings.tools

    if enabled is not None:
        deny = [t for t in policy.deny if t != name]
        allow = list(policy.allow)
        if enabled:
            # If an exclusive allow list is in force, the tool has to join it.
            if allow and name not in allow:
                allow.append(name)
        else:
            deny.append(name)
            allow = [t for t in allow if t != name]
        policy.deny = sorted(set(deny))
        policy.allow = sorted(set(allow))

    if auto_confirm is not None:
        current = [t for t in policy.auto_confirm if t != name]
        if auto_confirm:
            current.append(name)
        policy.auto_confirm = sorted(set(current))

    settings.tools = policy
    return save_assistant_settings(settings)
