"""SearXNG discovery — returns ranked candidates, not answers."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import httpx

from daemon.config import research_settings

logger = logging.getLogger(__name__)


def _canonical_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{host}{path}"


def _domain(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


async def search_searx(query: str, *, limit: int = 20) -> list[dict[str, str]]:
    cfg = research_settings()
    base = str(cfg.get("searxng_url", "http://127.0.0.1:8080")).rstrip("/")
    timeout = float(cfg.get("timeout_seconds", 25))

    params = {"q": query, "format": "json", "categories": "general"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("searxng search failed: %s", exc)
        return []

    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in data.get("results", []):
        url = str(item.get("url", "")).strip()
        if not url or url.lower().endswith(".pdf"):
            continue
        key = _canonical_url(url)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "title": str(item.get("title", "")).strip() or url,
                "url": url,
                "snippet": str(item.get("content", "")).strip(),
                "domain": _domain(url),
            }
        )
        if len(results) >= limit:
            break
    return results


def diversify_candidates(candidates: list[dict[str, str]], max_count: int) -> list[dict[str, str]]:
    """Pick diverse URLs — not just positional top-N from one domain."""
    picked: list[dict[str, str]] = []
    domain_counts: dict[str, int] = {}

    for item in candidates:
        dom = item.get("domain", "")
        if domain_counts.get(dom, 0) >= 2:
            continue
        picked.append(item)
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        if len(picked) >= max_count:
            break

    if len(picked) < max_count:
        for item in candidates:
            if item in picked:
                continue
            picked.append(item)
            if len(picked) >= max_count:
                break
    return picked


def merge_round_robin(
    result_sets: list[list[dict[str, str]]],
    *,
    limit: int,
) -> tuple[list[dict[str, str]], bool]:
    """Interleave several queries' results, best-of-each first.

    Concatenating result sets lets the first query fill the whole budget, so a
    second query that found the actual answer never gets read. Taking one result
    at each rank from every query before advancing gives each query a share.

    Returns the merged list and whether anything was dropped to honour `limit`.
    """
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    depth = max((len(rows) for rows in result_sets), default=0)

    for rank in range(depth):
        for rows in result_sets:
            if rank >= len(rows):
                continue
            key = _canonical_url(rows[rank]["url"])
            if key in seen:
                continue
            seen.add(key)
            if len(merged) < limit:
                merged.append(rows[rank])

    # Truncated means the limit dropped something, not that a URL was deduped:
    # telling the model to "refine the query for more" when there is no more is
    # advice that sends it in a circle.
    return merged, len(seen) > len(merged)
