"""Catalog install survives a registry rebuild (gateway restart)."""

from __future__ import annotations

from pathlib import Path

from app.config import AppConfig
from app.plugins.registry import PluginRegistry


MAILSPRING_TOML = """\
id = "mailspring"
name = "Mailspring"
kind = "mcp"
description = "Mail tools via MCP."
worker_id = ""
cloud = false
command = "npx"
args = ["--yes", "mcp-remote@latest", "http://127.0.0.1:2587/mcp"]
health_tool = "list_folders"
run_tool = "search_mail"
env = { AUTH_HEADER = "Bearer test" }
"""


def _gateway_root(tmp_path: Path) -> Path:
    root = tmp_path / "gateway"
    (root / "config").mkdir(parents=True)
    (root / "plugins" / "community").mkdir(parents=True)
    catalog = root / "plugins" / "catalog" / "mailspring"
    catalog.mkdir(parents=True)
    (catalog / "plugin.toml").write_text(MAILSPRING_TOML, encoding="utf-8")
    return root


def test_catalog_mcp_plugin_stays_loaded_after_restart(tmp_path: Path) -> None:
    root = _gateway_root(tmp_path)
    config = AppConfig(root=root)

    first = PluginRegistry(config)
    assert "mailspring" not in {p.id for p in first.list()}
    first.install_from_catalog("mailspring")
    assert "mailspring" in {p.id for p in first.list()}
    assert first.is_installed("mailspring")

    # Simulate a gateway restart: new registry, same on-disk state.
    second = PluginRegistry(config)
    assert second.is_installed("mailspring")
    assert "mailspring" in {p.id for p in second.list()}, (
        "catalog MCP plugins must reload from plugins/catalog after restart"
    )
    catalog = {e["id"]: e for e in second.catalog()}
    assert catalog["mailspring"]["installed"] is True


def test_write_state_keeps_installed_entry_if_load_fails(
    tmp_path: Path, monkeypatch
) -> None:
    root = _gateway_root(tmp_path)
    config = AppConfig(root=root)
    registry = PluginRegistry(config)
    registry.install_from_catalog("mailspring")

    # Pretend discovery cannot build the plugin, then save another plugin's state.
    registry._plugins.pop("mailspring", None)
    registry._write_state()

    raw = (root / "config" / "plugins.toml").read_text(encoding="utf-8")
    assert "[plugins.mailspring]" in raw
    assert "installed = true" in raw.lower() or "installed = true" in raw
