"""Tests for Keylane greenfield."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agent.tools_parse import parse_tool_call, strip_tool_call
from daemon.config import (
    add_mcp_server,
    all_settings,
    list_mcp_servers,
    remove_mcp_server,
    reset_settings,
    save_settings,
)
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


def test_parse_tool_call_nested_arguments():
    text = (
        "<tool_call>\n"
        '{"name": "research_web", "arguments": {"question": "siege weapons civ6"}}\n'
        "</tool_call>"
    )
    call = parse_tool_call(text)
    assert call is not None
    assert call["name"] == "research_web"
    assert call["arguments"]["question"] == "siege weapons civ6"


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


def test_a_stdio_server_keeps_its_arguments_and_environment(isolated_settings, monkeypatch):
    monkeypatch.setattr("daemon.config._config_mcp_servers", lambda: [])
    add_mcp_server(
        {
            "id": "fs",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me"],
            "env": {"TOKEN": "abc"},
        }
    )
    saved = next(s for s in list_mcp_servers() if s["id"] == "fs")
    assert saved["args"] == ["-y", "@modelcontextprotocol/server-filesystem", "/home/me"]
    assert saved["env"] == {"TOKEN": "abc"}


def test_an_http_server_keeps_its_url_and_token(isolated_settings, monkeypatch):
    """The Mailspring shape: a URL and a token, no command anywhere."""
    monkeypatch.setattr("daemon.config._config_mcp_servers", lambda: [])
    add_mcp_server(
        {
            "id": "mailspring",
            "transport": "http",
            "url": "http://127.0.0.1:2587/mcp",
            "auth_header": "abc-123",
            "headers": {"X-Trace": "1"},
        }
    )
    saved = next(s for s in list_mcp_servers() if s["id"] == "mailspring")
    assert saved["url"] == "http://127.0.0.1:2587/mcp"
    assert saved["auth_header"] == "abc-123"
    assert saved["headers"] == {"X-Trace": "1"}
    assert "command" not in saved


def test_an_http_server_without_a_url_is_refused(isolated_settings, monkeypatch):
    monkeypatch.setattr("daemon.config._config_mcp_servers", lambda: [])
    with pytest.raises(ValueError):
        add_mcp_server({"id": "broken", "transport": "http"})


def test_a_tools_schema_is_read_off_the_sdk_model():
    """The regression that silently emptied every MCP server.

    ``mcp`` 2.0 renamed this attribute to ``input_schema``; ``inputSchema``
    survives only as a wire alias, so reading it off the model raises. Building
    the Tool through the SDK is the point of the test — a hand-rolled stub with
    both names spelled out would pass no matter which one the code reads.
    """
    from mcp.types import Tool as SdkTool
    from mcpbridge.client import tool_input_schema

    schema = {"type": "object", "properties": {"folderId": {"type": "string"}}}
    tool = SdkTool(name="list_threads", description="", inputSchema=schema)

    assert tool_input_schema(tool) == schema


def test_a_server_registers_every_tool_it_offers(monkeypatch):
    """Registration used to abort on the first tool and report nothing wrong."""
    import asyncio

    from mcp.types import Tool as SdkTool

    from mcpbridge import client as mcp_client
    from tools.registry import ToolRegistry

    offered = [
        SdkTool(name=f"tool_{i}", description="", inputSchema={"type": "object"})
        for i in range(3)
    ]

    class _Session:
        async def list_tools(self):
            return type("Result", (), {"tools": offered})()

    monkeypatch.setattr(
        mcp_client, "mcp_settings", lambda: {"servers": [{"id": "mail", "transport": "http"}]}
    )
    monkeypatch.setattr(mcp_client, "_is_disabled", lambda name: False)

    async def _session(sid, srv):
        return _Session()

    monkeypatch.setattr(mcp_client, "_get_session", _session)

    registry = ToolRegistry()
    count = asyncio.run(mcp_client.load_mcp_tools(registry))

    assert count == 3
    assert sorted(registry._tools) == ["mcp.mail.tool_0", "mcp.mail.tool_1", "mcp.mail.tool_2"]


# ── one transcript, balanced windows ─────────────────────────────────────


def test_a_leading_tool_result_is_dropped_from_a_window():
    """Its assistant call fell off the front, so it answers nothing."""
    from memory.store import balance_tool_pairs

    window = [
        {"role": "user", "content": '<tool_result name="recall">\n[]\n</tool_result>'},
        {"role": "assistant", "content": "You have no meetings."},
    ]
    assert balance_tool_pairs(window) == window[1:]


def test_a_trailing_tool_call_without_its_result_is_dropped():
    from memory.store import balance_tool_pairs

    window = [
        {"role": "user", "content": "any meetings?"},
        {"role": "assistant", "content": '{"tool_call": "recall", "arguments": {}}'},
    ]
    assert balance_tool_pairs(window) == window[:1]


def test_a_balanced_window_is_untouched():
    from memory.store import balance_tool_pairs

    window = [
        {"role": "user", "content": "any meetings?"},
        {"role": "assistant", "content": '{"tool_call": "recall", "arguments": {}}'},
        {"role": "user", "content": '<tool_result name="recall">\n[]\n</tool_result>'},
        {"role": "assistant", "content": "None today."},
    ]
    assert balance_tool_pairs(window) == window


def test_the_tool_result_block_is_the_only_rendering():
    """The model and the session log must see the same string."""
    from agent.loop import tool_result_block

    block = tool_result_block("recall", "[]")
    assert block == '<tool_result name="recall">\n[]\n</tool_result>'
    assert "Done." in tool_result_block("remember", "ok", note="Done.")


# ── retrieval scoring ────────────────────────────────────────────────────


def test_bm25_ranks_across_the_candidate_set():
    from research.provider import bm25_scores

    docs = [
        "OpenVINO Intel NPU performance benchmarks on Linux",
        "recipe for chocolate cake baking tips",
        "Intel NPU driver notes",
    ]
    scores = bm25_scores("intel npu openvino performance", docs)
    assert scores[0] > scores[2] > scores[1] == 0.0


def test_a_term_common_to_every_candidate_carries_less_weight():
    """This is what an in-document frequency stand-in for idf cannot express."""
    from research.provider import bm25_scores

    common = ["intel npu notes", "intel npu guide", "intel npu faq"]
    distinctive = ["intel npu notes", "intel npu guide", "intel npu openvino"]
    assert bm25_scores("openvino", distinctive)[2] > 0
    assert max(bm25_scores("intel", common)) < bm25_scores("openvino", distinctive)[2]


def test_coverage_score_is_an_absolute_scale():
    """The relevance gates compare against fixed thresholds, so they need one."""
    from research.provider import coverage_score

    # Two terms, three hits each: 3/(3+1.5) = 0.67 per term.
    assert coverage_score("intel npu", "intel npu intel npu intel npu") == pytest.approx(0.667, abs=0.01)
    assert coverage_score("intel npu", "intel npu") == pytest.approx(0.4, abs=0.01)
    assert coverage_score("intel npu", "chocolate cake") == 0.0
    assert 0.0 <= coverage_score("intel npu openvino", "intel notes") <= 1.0


def test_the_spotlight_is_centred_on_screen():
    from ui.placement import floating_geometry

    x, y = floating_geometry("spotlight", 600, 400, (1920, 1080), 24)
    assert (x, y) == ((1920 - 600) // 2, (1080 - 400) // 2)


def test_the_corner_panel_stays_in_the_corner():
    from ui.placement import floating_geometry

    x, y = floating_geometry("corner", 400, 300, (1920, 1080), 24)
    assert (x, y) == (1920 - 400 - 24, 24)


def test_centering_clamps_to_the_screen():
    from ui.placement import centered_geometry

    assert centered_geometry(3000, 2000, (1920, 1080)) == (0, 0)


def test_scaling_a_rectangle_scales_both_position_and_size():
    """wmctrl wants device pixels for both, whatever its readback claims.

    `wmctrl -lG` reports a doubled position, which once persuaded me the
    position should go out unscaled. It should not: dropping the scale factor
    moves every window to half its offset, into the top-left corner.
    """
    from ui.placement import scaled_geometry

    assert scaled_geometry(600, 310, 720, 580, 2) == (1200, 620, 1440, 1160)
    assert scaled_geometry(600, 310, 720, 580, 1) == (600, 310, 720, 580)


def test_a_centred_window_has_equal_gaps_on_the_device():
    """Centred stays centred once everything is in device pixels."""
    from ui.placement import centered_geometry, scaled_geometry

    logical_screen = (1920, 1200)
    width, height = 720, 580
    for scale in (1, 2, 3):
        x, y = centered_geometry(width, height, logical_screen)
        px, py, pw, ph = scaled_geometry(x, y, width, height, scale)
        screen_w, screen_h = (v * scale for v in logical_screen)
        assert abs((screen_w - pw) - 2 * px) <= scale
        assert abs((screen_h - ph) - 2 * py) <= scale
