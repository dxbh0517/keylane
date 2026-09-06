"""Settings and subsystem health checks."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from daemon.config import mcp_settings, research_settings
from models.catalog import get_runtime

logger = logging.getLogger(__name__)


async def check_searxng() -> dict[str, Any]:
    cfg = research_settings()
    base = str(cfg.get("searxng_url", "http://127.0.0.1:8080")).rstrip("/")
    timeout = float(cfg.get("timeout_seconds", 25))
    try:
        async with httpx.AsyncClient(timeout=min(timeout, 10)) as client:
            resp = await client.get(f"{base}/search", params={"q": "test", "format": "json"})
            resp.raise_for_status()
            count = len(resp.json().get("results", []))
            return {"ok": True, "url": base, "sample_results": count}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": base, "error": str(exc)}


async def check_mcp_servers() -> list[dict[str, Any]]:
    from mcpbridge.client import probe_mcp_server

    servers = mcp_settings().get("servers", [])
    out: list[dict[str, Any]] = []
    for srv in servers:
        sid = srv.get("id", "mcp")
        try:
            info = await probe_mcp_server(srv)
            out.append({"id": sid, "ok": True, **info})
        except Exception as exc:  # noqa: BLE001
            out.append({"id": sid, "ok": False, "error": str(exc)})
    return out


def check_screen_layer() -> dict[str, Any]:
    """Whether dictation and annotation can work on this machine.

    Both depend on binaries rather than on configuration, and both fail in ways
    that look like a Keylane bug from the outside — nothing typed, nothing
    drawn. Reporting them beside SearXNG and MCP is what makes the cause
    visible before the user files it as one.
    """
    from vision import ocr

    try:
        from inject import describe as describe_injection

        injection = describe_injection()
    except Exception as exc:  # noqa: BLE001
        injection = {"ok": False, "error": str(exc), "available": []}

    return {
        "injection": injection,
        "ocr": {
            "ok": ocr.available(),
            "reason": "" if ocr.available() else "tesseract is not installed",
        },
    }


def check_prompt_budget() -> dict[str, Any]:
    """How much of the prompt budget the conversation actually gets.

    A prompt that has crowded out the conversation does not fail — it answers,
    without history and without whatever guidance was dropped to fit. That is
    how twenty-one working MCP tools looked like a broken MCP server, so the
    number is reported next to the things that do fail loudly.
    """
    from seams import get_context
    from tools.registry import get_registry

    ctx = get_context()
    budget = ctx.llm.prompt_budget_chars("interactive")
    if budget <= 0:
        # No model resident, so nothing has imposed a limit yet. Saying "100%
        # free" here would be a fiction, and a reassuring one.
        return {"ok": True, "budget_chars": 0, "reason": "no model loaded yet"}
    try:
        assembly = ctx.prompt.assemble(budget_chars=budget)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    tools = len(get_registry().visible())
    return {
        "ok": not assembly.starved,
        "budget_chars": budget,
        "headroom_chars": max(assembly.headroom_chars, 0),
        "headroom_percent": round(assembly.headroom_share * 100),
        "tools": tools,
        "reason": (
            ""
            if not assembly.starved
            else (
                f"{tools} tools leave only {max(assembly.headroom_chars, 0)} of "
                f"{budget} characters for the conversation. Disable tools you do "
                "not use, or raise KEYLANE_NPU_MAX_PROMPT_TOKENS."
            )
        ),
    }


async def settings_health() -> dict[str, Any]:
    searx = await check_searxng()
    mcp = await check_mcp_servers()
    runtime = get_runtime()
    cfg = research_settings()
    from seams import get_context

    ctx = get_context()
    return {
        "npu": runtime.status,
        # Which adapter each purpose currently resolves to — the answer to
        # "is background work actually on the GPU model?"
        "models": ctx.llm.status(),
        "subagents": ctx.subagents.status(),
        "searxng": searx,
        "mcp": mcp,
        "screen": check_screen_layer(),
        "prompt": check_prompt_budget(),
        "research": {
            "search_backend": cfg.get("search_backend", "searxng"),
            "extract_backend": cfg.get("extract_backend", "local"),
            "keyless_fallback": cfg.get("keyless_fallback", True),
        },
    }
