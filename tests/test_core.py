"""Tests for Keylane greenfield."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agent.tools_parse import parse_tool_call, strip_tool_call
from daemon.config import add_mcp_server, all_settings, remove_mcp_server, reset_settings, save_settings
from research.search import diversify_candidates
from research.provider import bm25_score
from research.researcher import _chunk_text, _compress_evidence, Source


@pytest.fixture()
def isolated_settings(monkeypatch, tmp_path: Path):
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr("daemon.config.SETTINGS_PATH", settings_file)
    reset_settings(None)
    yield settings_file
    reset_settings(None)


def test_parse_tool_call():
    text = 'Hello <tool_call>\n{"name": "research_web", "arguments": {"question": "test"}}\n</tool_call>'
    call = parse_tool_call(text)
    assert call is not None
    assert call["name"] == "research_web"
    assert call["arguments"]["question"] == "test"
    assert "tool_call" not in strip_tool_call(text)


def test_parse_function_call_json():
    text = '{"name": "web_search", "arguments": {"query": "fedora 44"}}'
    call = parse_tool_call(text)
    assert call is not None
    assert call["name"] == "web_search"


def test_diversify_candidates_not_top_five_only():
    candidates = [
        {"url": f"https://a.com/{i}", "domain": "a.com", "title": f"A{i}", "snippet": ""}
        for i in range(5)
    ] + [
        {"url": "https://b.com/best", "domain": "b.com", "title": "Best answer", "snippet": "relevant"},
        {"url": "https://c.com/other", "domain": "c.com", "title": "Other", "snippet": ""},
    ]
    picked = diversify_candidates(candidates, 4)
    domains = {c["domain"] for c in picked}
    assert "b.com" in domains
    assert len(picked) == 4
    assert picked[0]["domain"] == "a.com"


def test_settings_merge_and_save(isolated_settings):
    save_settings("assistant", {"name": "Jarvis"})
    data = all_settings()
    assert data["assistant"]["name"] == "Jarvis"
    assert "iteration_budget" in data["assistant"]


def test_settings_reset_section(isolated_settings):
    save_settings("research", {"search_backend": "ddgs"})
    reset_settings("research")
    data = all_settings()
    assert data["research"]["search_backend"] == "searxng"


def test_bm25_scores_relevant_higher():
    q = "intel npu openvino performance"
    good = bm25_score(q, "OpenVINO Intel NPU performance benchmarks on Linux")
    bad = bm25_score(q, "recipe for chocolate cake baking tips")
    assert good > bad


def test_evidence_compression():
    pages = [{"text": ("intel npu openvino benchmark data " * 30).strip()}]
    sources = [Source(1, "Doc", "https://example.com")]
    blob = _compress_evidence("intel npu openvino", pages, sources)
    assert "[1]" in blob
    assert "intel" in blob.lower()


def test_chunk_text():
    text = "paragraph one. " * 80
    chunks = _chunk_text(text, chunk_size=200)
    assert len(chunks) >= 2


def test_registry_maps_query_to_question():
    from tools.builtin import register_builtin_tools
    from tools.registry import get_registry
    import asyncio

    register_builtin_tools()
    reg = get_registry()

    async def _run():
        tool = reg.get("research_web")
        assert tool is not None
        args = reg._normalize_arguments(tool, {"query": "test"})  # noqa: SLF001
        assert args["question"] == "test"

    asyncio.run(_run())


def test_settings_ui_theme(isolated_settings):
    save_settings("ui", {"theme": "dark"})
    data = all_settings()
    assert data["ui"]["theme"] == "dark"


def test_mcp_server_add(isolated_settings, monkeypatch):
    monkeypatch.setattr("daemon.config._config_mcp_servers", lambda: [])
    servers = add_mcp_server(
        {"id": "test", "command": "echo", "args": ["hello"], "transport": "stdio"}
    )
    assert any(s["id"] == "test" for s in servers)
    servers = remove_mcp_server("test")
    assert not any(s["id"] == "test" for s in servers)
