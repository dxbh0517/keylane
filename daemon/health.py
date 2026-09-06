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
        "research": {
            "search_backend": cfg.get("search_backend", "searxng"),
            "extract_backend": cfg.get("extract_backend", "local"),
            "keyless_fallback": cfg.get("keyless_fallback", True),
        },
    }
