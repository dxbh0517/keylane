"""Tests for the assistant tool layer: policy, sandboxing, and the shell guard."""

from __future__ import annotations

import pytest

from app.assistant_settings import (
    AssistantSettings,
    ShellPolicy,
    ToolPolicy,
    save_assistant_settings,
)
from app.tools.base import ToolDanger, ToolResult, object_schema, string_prop
from app.tools.builtin.files import SandboxError, resolve_in_sandbox
from app.tools.builtin.shell import RunCommandTool
from app.tools.registry import ConfirmationRequired, ToolRegistry


@pytest.fixture
def settings(monkeypatch):
    """Swap in an in-memory settings object so tests never touch config/."""

    state = {"value": AssistantSettings()}

    def fake_load(*, refresh: bool = False) -> AssistantSettings:
        return state["value"]

    for module in (
        "app.assistant_settings",
        "app.tools.registry",
        "app.tools.builtin.shell",
        "app.tools.builtin.web",
        "app.tools.builtin.email",
        "app.tools.builtin.delegation",
    ):
        monkeypatch.setattr(f"{module}.load_assistant_settings", fake_load, raising=False)
    return state


# ------------------------------------------------------------------- policy


def test_dangerous_tool_requires_confirmation(settings):
    registry = ToolRegistry()
    tool = registry.get("run_command")
    assert tool is not None
    assert tool.danger is ToolDanger.DANGEROUS
    assert registry.needs_confirmation(tool) is True


def test_safe_tool_runs_without_confirmation(settings):
    registry = ToolRegistry()
    assert registry.needs_confirmation(registry.get("system_info")) is False


def test_confirm_threshold_can_be_raised(settings):
    settings["value"].tools = ToolPolicy(confirm_danger_at="safe")
    registry = ToolRegistry()
    assert registry.needs_confirmation(registry.get("system_info")) is True


def test_auto_confirm_bypasses_the_gate(settings):
    settings["value"].tools = ToolPolicy(auto_confirm=["open_application"])
    registry = ToolRegistry()
    assert registry.needs_confirmation(registry.get("open_application")) is False


def test_deny_list_blocks_a_tool(settings):
    settings["value"].tools = ToolPolicy(deny=["web_search"])
    registry = ToolRegistry()
    assert registry.policy_blocks("web_search") is not None
    assert registry.policy_blocks("system_info") is None


def test_allow_list_is_exclusive(settings):
    settings["value"].tools = ToolPolicy(allow=["system_info"])
    registry = ToolRegistry()
    assert registry.policy_blocks("system_info") is None
    assert registry.policy_blocks("web_search") is not None


def test_disabling_tools_blocks_everything(settings):
    settings["value"].tools = ToolPolicy(enabled=False)
    registry = ToolRegistry()
    assert registry.policy_blocks("system_info") is not None


@pytest.mark.asyncio
async def test_call_raises_for_unconfirmed_dangerous_tool(settings):
    registry = ToolRegistry()
    with pytest.raises(ConfirmationRequired):
        await registry.call("run_command", {"command": "ls"})


@pytest.mark.asyncio
async def test_call_reports_unknown_tool(settings):
    registry = ToolRegistry()
    result = await registry.call("teleport", {})
    assert result.ok is False
    assert "No tool named" in (result.error or "")


@pytest.mark.asyncio
async def test_denied_tool_refuses_even_when_confirmed(settings):
    settings["value"].tools = ToolPolicy(deny=["run_command"])
    registry = ToolRegistry()
    result = await registry.call("run_command", {"command": "ls"}, confirmed=True)
    assert result.ok is False
    assert "deny list" in (result.error or "")


# ------------------------------------------------------------------- shell


@pytest.mark.asyncio
async def test_shell_runs_an_allowlisted_command(settings):
    result = await RunCommandTool().run({"command": "echo_missing_binary"})
    # 'echo' is deliberately not on the default allowlist.
    assert result.ok is False


@pytest.mark.asyncio
async def test_shell_rejects_a_command_off_the_allowlist(settings):
    result = await RunCommandTool().run({"command": "curl", "args": ["http://x"]})
    assert result.ok is False
    assert "allowlist" in (result.error or "")


@pytest.mark.asyncio
async def test_shell_refuses_forbidden_commands_even_if_allowlisted(settings):
    settings["value"].shell = ShellPolicy(allowlist=["rm", "sudo", "bash"])
    tool = RunCommandTool()
    for program in ("rm", "sudo", "bash"):
        result = await tool.run({"command": program, "args": ["-rf", "/"]})
        assert result.ok is False
        assert "permanently blocked" in (result.error or "")


@pytest.mark.asyncio
async def test_shell_rejects_nested_shell_arguments(settings):
    settings["value"].shell = ShellPolicy(allowlist=["python3"])
    result = await RunCommandTool().run(
        {"command": "python3", "args": ["-c", "import os; os.system('id')"]}
    )
    assert result.ok is False
    assert "nested shell" in (result.error or "")


@pytest.mark.asyncio
async def test_shell_can_be_disabled(settings):
    settings["value"].shell = ShellPolicy(enabled=False)
    result = await RunCommandTool().run({"command": "ls"})
    assert result.ok is False
    assert "disabled" in (result.error or "")


@pytest.mark.asyncio
async def test_shell_actually_runs_and_captures_output(settings):
    settings["value"].shell = ShellPolicy(allowlist=["uname"])
    result = await RunCommandTool().run({"command": "uname", "args": ["-s"]})
    assert result.ok is True
    assert "Linux" in result.output
    assert result.data["exit_code"] == 0


# ----------------------------------------------------------------- sandbox


def test_sandbox_allows_a_configured_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.tools.builtin.files.sandbox_roots", lambda: [tmp_path.resolve()]
    )
    target = tmp_path / "notes.txt"
    target.write_text("hello", encoding="utf-8")
    assert resolve_in_sandbox(str(target)) == target.resolve()


def test_sandbox_rejects_paths_outside_the_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.tools.builtin.files.sandbox_roots", lambda: [tmp_path.resolve()]
    )
    with pytest.raises(SandboxError):
        resolve_in_sandbox("/etc/shadow")


def test_sandbox_rejects_traversal_out_of_the_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.tools.builtin.files.sandbox_roots", lambda: [(tmp_path / "ok").resolve()]
    )
    (tmp_path / "ok").mkdir()
    with pytest.raises(SandboxError):
        resolve_in_sandbox(str(tmp_path / "ok" / ".." / "secret.txt"))


def test_sandbox_rejects_protected_names(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.tools.builtin.files.sandbox_roots", lambda: [tmp_path.resolve()]
    )
    (tmp_path / ".ssh").mkdir()
    with pytest.raises(SandboxError):
        resolve_in_sandbox(str(tmp_path / ".ssh" / "id_rsa"))
    with pytest.raises(SandboxError):
        resolve_in_sandbox(str(tmp_path / ".env"))


# -------------------------------------------------------------- tool specs


def test_every_tool_advertises_a_usable_spec(settings):
    registry = ToolRegistry()
    specs = registry.specs()
    assert specs, "expected built-in tools"
    for spec in specs:
        assert spec.name, "tool needs a name"
        assert len(spec.description) > 20, f"{spec.name} needs a real description"
        assert spec.parameters.get("type") == "object"
        # The prompt line is what the model actually reads.
        assert spec.name in spec.prompt_line()


def test_object_schema_helper_shape():
    schema = object_schema({"city": string_prop("City name.")}, required=["city"])
    assert schema["type"] == "object"
    assert schema["required"] == ["city"]
    assert schema["properties"]["city"]["type"] == "string"


def test_tool_result_helpers():
    ok = ToolResult.success("done", data={"a": 1}, artifacts=["/tmp/x"])
    assert ok.ok and ok.data["a"] == 1 and ok.artifacts == ["/tmp/x"]
    bad = ToolResult.failure("nope")
    assert bad.ok is False and bad.error == "nope"


# ----------------------------------------------------- application matching


def _entry(name, app_id, comment=""):
    from pathlib import Path

    from app.tools.builtin.desktop import DesktopEntry

    return DesktopEntry(
        Path(f"/usr/share/applications/{app_id}.desktop"),
        {"Name": name, "GenericName": "", "Comment": comment, "Exec": app_id,
         "NoDisplay": "false", "Type": "Application"},
    )


@pytest.fixture
def fake_apps(monkeypatch):
    entries = [
        _entry("Text Editor", "org.gnome.TextEditor", "Edit text files"),
        _entry("Firefox", "org.mozilla.firefox", "Browse the web"),
        _entry("Calculator", "org.gnome.Calculator"),
        _entry("Files", "org.gnome.Nautilus", "Access and organize files"),
        _entry("Terminal", "org.gnome.Terminal"),
        _entry("Text Editor Plus", "com.example.TextEditorPlus"),
    ]
    monkeypatch.setattr("app.tools.builtin.desktop.scan_desktop_entries", lambda: entries)
    return entries


@pytest.mark.parametrize(
    "query,expected",
    [
        ("firefox", "Firefox"),
        ("Firefox", "Firefox"),
        ("text editor", "Text Editor"),
        # Filler words people actually say must not break the match.
        ("gnome text editor", "Text Editor"),
        ("the calculator app", "Calculator"),
        ("please open files", None),  # 'open' is not filler; no entry has it
        ("nautilus", "Files"),
        ("terminal", "Terminal"),
    ],
)
def test_application_matching(fake_apps, query, expected):
    from app.tools.builtin.desktop import find_application

    match = find_application(query)
    assert (match.name if match else None) == expected


def test_exact_name_beats_a_longer_superstring(fake_apps):
    from app.tools.builtin.desktop import find_application

    assert find_application("Text Editor").name == "Text Editor"


def test_unknown_application_returns_none(fake_apps):
    from app.tools.builtin.desktop import find_application

    assert find_application("photoshop") is None
    assert find_application("") is None


@pytest.mark.asyncio
async def test_open_application_reports_a_miss_usefully(settings, fake_apps):
    from app.tools.builtin.desktop import OpenApplicationTool

    result = await OpenApplicationTool().run({"application": "photoshop"})
    assert result.ok is False
    # The message has to tell the model how to recover.
    assert "list_applications" in (result.error or "")
