"""Web tools — search the web and read a page as text."""

from __future__ import annotations

import html
import logging
import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from app.assistant_settings import load_assistant_settings
from app.tools.base import (
    BaseTool,
    ToolDanger,
    ToolResult,
    int_prop,
    object_schema,
    string_prop,
)

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) Keylane/1.0 (local assistant)"

_TAG = re.compile(r"<[^>]+>")
_SCRIPT_STYLE = re.compile(
    r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANKLINES = re.compile(r"\n{3,}")


def strip_html(markup: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", markup)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", text, flags=re.IGNORECASE)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    text = _WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANKLINES.sub("\n\n", text).strip()


def _unwrap_duckduckgo(href: str) -> str:
    """DuckDuckGo HTML results wrap targets in /l/?uddg=<encoded>."""
    if "uddg=" not in href:
        return href if href.startswith("http") else f"https:{href}" if href.startswith("//") else href
    query = urlparse(href if href.startswith("http") else f"https://duckduckgo.com{href}").query
    values = parse_qs(query).get("uddg")
    return unquote(values[0]) if values else href


async def _duckduckgo(query: str, limit: int, timeout: float) -> list[dict[str, str]]:
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        response = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
        )
        response.raise_for_status()
        markup = response.text

    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippets = re.findall(
        r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
        markup,
        re.IGNORECASE | re.DOTALL,
    )
    for index, match in enumerate(pattern.finditer(markup)):
        if len(results) >= limit:
            break
        url = _unwrap_duckduckgo(match.group(1))
        title = strip_html(match.group(2))
        snippet = strip_html(snippets[index]) if index < len(snippets) else ""
        if not url.startswith("http"):
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
    return results


async def _searxng(query: str, limit: int, timeout: float, base_url: str) -> list[dict[str, str]]:
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        response = await client.get(
            f"{base_url.rstrip('/')}/search",
            params={"q": query, "format": "json"},
        )
        response.raise_for_status()
        payload = response.json()
    results = []
    for item in (payload.get("results") or [])[:limit]:
        results.append(
            {
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "snippet": str(item.get("content") or ""),
            }
        )
    return results


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the web and return titles, URLs and snippets. Use it for current "
        "events, documentation lookups, prices, or anything outside your knowledge."
    )
    danger = ToolDanger.SAFE
    category = "web"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "query": string_prop("Search terms."),
                "limit": int_prop("Number of results (default from settings).", default=5),
            },
            required=["query"],
        )

    def availability(self) -> str | None:
        settings = load_assistant_settings()
        if settings.search.engine == "none":
            return "Web search is disabled in assistant settings"
        return None

    async def run(self, args: dict[str, Any]) -> ToolResult:
        settings = load_assistant_settings().search
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult.failure("No search query given.")
        limit = max(1, min(int(args.get("limit") or settings.max_results), 15))

        try:
            if settings.engine == "searxng":
                results = await _searxng(
                    query, limit, settings.timeout_seconds, settings.searxng_url
                )
            elif settings.engine == "none":
                return ToolResult.failure("Web search is disabled in assistant settings.")
            else:
                results = await _duckduckgo(query, limit, settings.timeout_seconds)
        except httpx.HTTPError as exc:
            return ToolResult.failure(f"Search request failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Web search failed")
            return ToolResult.failure(f"Search failed: {exc}")

        if not results:
            return ToolResult.success(
                f"No results for '{query}'.", data={"query": query, "results": []}
            )

        lines = [
            f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}"
            for i, r in enumerate(results, start=1)
        ]
        return ToolResult.success(
            "\n".join(lines), data={"query": query, "results": results}
        )


class FetchUrlTool(BaseTool):
    name = "fetch_url"
    description = (
        "Download a web page and return its readable text. Use after web_search "
        "when a snippet is not enough."
    )
    danger = ToolDanger.SAFE
    category = "web"

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "url": string_prop("Absolute http(s) URL to fetch."),
                "max_chars": int_prop("Truncate the text at this length (default 6000).", default=6000),
            },
            required=["url"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        url = str(args.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            return ToolResult.failure("url must start with http:// or https://")
        max_chars = max(500, min(int(args.get("max_chars") or 6000), 40000))
        timeout = load_assistant_settings().search.timeout_seconds
        try:
            async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                body = response.text
        except httpx.HTTPError as exc:
            return ToolResult.failure(f"Could not fetch {url}: {exc}")

        text = strip_html(body) if "html" in content_type or body.lstrip().startswith("<") else body
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "\n…[truncated]"
        title_match = re.search(r"<title[^>]*>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
        title = strip_html(title_match.group(1)) if title_match else url
        return ToolResult.success(
            text,
            data={"url": url, "title": title, "truncated": truncated, "chars": len(text)},
        )


def web_tools() -> list[BaseTool]:
    return [WebSearchTool(), FetchUrlTool()]
