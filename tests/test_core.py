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
