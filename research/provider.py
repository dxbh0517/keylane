"""Pluggable web search and extract providers."""

from __future__ import annotations

import abc
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from daemon.config import research_settings

logger = logging.getLogger(__name__)

_search_cache: dict[str, tuple[float, list[dict[str, str]]]] = {}


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    domain: str = ""


class WebSearchProvider(abc.ABC):
    name: str

    @abc.abstractmethod
    async def search(self, query: str, *, limit: int = 15) -> list[SearchResult]:
        raise NotImplementedError


class WebExtractProvider(abc.ABC):
    name: str

    @abc.abstractmethod
    async def extract(self, url: str) -> dict[str, str]:
        raise NotImplementedError


class SearxngProvider(WebSearchProvider):
    name = "searxng"

    async def search(self, query: str, *, limit: int = 15) -> list[SearchResult]:
        from research.search import search_searx

        rows = await search_searx(query, limit=limit)
        return [
            SearchResult(
                title=r["title"],
                url=r["url"],
                snippet=r.get("snippet", ""),
                domain=r.get("domain", ""),
            )
            for r in rows
        ]


class DdgsProvider(WebSearchProvider):
    name = "ddgs"

    async def search(self, query: str, *, limit: int = 15) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore[no-redef]
            except ImportError as exc:
                raise RuntimeError("ddgs not installed; pip install ddgs") from exc

        def _run() -> list[SearchResult]:
            out: list[SearchResult] = []
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=limit):
                    url = str(item.get("href") or item.get("url") or "").strip()
                    if not url:
                        continue
                    from research.search import _domain

                    out.append(
                        SearchResult(
                            title=str(item.get("title", url)),
                            url=url,
                            snippet=str(item.get("body") or item.get("snippet") or ""),
                            domain=_domain(url),
                        )
                    )
            return out

        import asyncio

        return await asyncio.to_thread(_run)


class LocalExtractProvider(WebExtractProvider):
    name = "local"

    async def extract(self, url: str) -> dict[str, str]:
        from research.fetch import fetch_page

        return await fetch_page(url)


def _cache_key(query: str, backend: str) -> str:
    return f"{backend}:{query.strip().lower()}"


def _cache_get(key: str) -> list[dict[str, str]] | None:
    cfg = research_settings()
    ttl = int(cfg.get("cache_ttl_minutes", 15)) * 60
    entry = _search_cache.get(key)
    if not entry:
        return None
    ts, rows = entry
    if time.time() - ts > ttl:
        del _search_cache[key]
        return None
    return rows


def _cache_put(key: str, rows: list[dict[str, str]]) -> None:
    _search_cache[key] = (time.time(), rows)


def _to_dicts(results: list[SearchResult]) -> list[dict[str, str]]:
    return [
        {
            "title": r.title,
            "url": r.url,
            "snippet": r.snippet,
            "domain": r.domain,
        }
        for r in results
    ]


def get_search_provider(name: str | None = None) -> WebSearchProvider:
    backend = name or research_settings().get("search_backend", "searxng")
    if backend == "ddgs":
        return DdgsProvider()
    return SearxngProvider()


def get_extract_provider(name: str | None = None) -> WebExtractProvider:
    backend = name or research_settings().get("extract_backend", "local")
    if backend != "local":
        logger.debug("unknown extract backend %s, using local", backend)
    return LocalExtractProvider()


async def search_with_fallback(query: str, *, limit: int = 15) -> list[dict[str, str]]:
    cfg = research_settings()
    primary = str(cfg.get("search_backend", "searxng"))
    key = _cache_key(query, primary)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    providers: list[WebSearchProvider] = [get_search_provider(primary)]
    if cfg.get("keyless_fallback", True) and primary != "ddgs":
        providers.append(DdgsProvider())

    last_err: Exception | None = None
    for provider in providers:
        try:
            results = _to_dicts(await provider.search(query, limit=limit))
            if results:
                _cache_put(_cache_key(query, provider.name), results)
                return results
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("search provider %s failed: %s", provider.name, exc)

    if last_err:
        logger.error("all search providers failed: %s", last_err)
    return []


def bm25_score(query: str, document: str, *, k1: float = 1.5, b: float = 0.75) -> float:
    """Simple BM25 over tokenized query/document."""
    q_tokens = [w.lower() for w in query.split() if len(w) > 2]
    if not q_tokens:
        return 0.0
    doc_tokens = [w.lower() for w in document.split()]
    if not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    avgdl = max(doc_len, 120)
    tf = Counter(doc_tokens)
    score = 0.0
    for term in set(q_tokens):
        freq = tf.get(term, 0)
        if freq == 0:
            continue
        idf = math.log(1 + 1.0 / (1 + freq))
        denom = freq + k1 * (1 - b + b * doc_len / avgdl)
        score += idf * (freq * (k1 + 1)) / denom
    return score
