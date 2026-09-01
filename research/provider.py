"""Pluggable web search and extract providers."""

from __future__ import annotations

import abc
import logging
import math
import re
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


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 2]


def bm25_scores(
    query: str,
    documents: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Score every document against the query with BM25 over that set.

    Inverse *document* frequency needs a corpus, so the ranked set is the
    corpus: a term appearing in one of twenty candidates says far more about
    that candidate than a term appearing in all twenty. Scoring documents one
    at a time cannot express that, which is why the whole set is scored here.
    """
    q_terms = set(_tokens(query))
    if not q_terms or not documents:
        return [0.0] * len(documents)

    tokenized = [_tokens(doc) for doc in documents]
    lengths = [len(t) for t in tokenized]
    non_empty = [n for n in lengths if n]
    avgdl = (sum(non_empty) / len(non_empty)) if non_empty else 1.0
    total_docs = len(documents)

    doc_freq: dict[str, int] = {}
    counters = [Counter(t) for t in tokenized]
    for term in q_terms:
        doc_freq[term] = sum(1 for c in counters if c.get(term))

    scores: list[float] = []
    for counter, length in zip(counters, lengths):
        if not length:
            scores.append(0.0)
            continue
        score = 0.0
        for term in q_terms:
            freq = counter.get(term, 0)
            if not freq:
                continue
            n_q = doc_freq[term]
            # Robertson/Sparck-Jones idf, smoothed so it never goes negative.
            idf = math.log(1 + (total_docs - n_q + 0.5) / (n_q + 0.5))
            denom = freq + k1 * (1 - b + b * length / avgdl)
            score += idf * (freq * (k1 + 1)) / denom
        scores.append(score)
    return scores


def coverage_score(query: str, document: str) -> float:
    """How much of the query this one document covers, in ``[0, 1]``.

    BM25 is a *ranking* function: its magnitude depends on the corpus, so it
    cannot be compared against a fixed threshold. The relevance gates in the
    research pipeline need an absolute scale, so they use this instead — the
    share of distinct query terms present, weighted by a saturating count so a
    page that mentions a term once is not treated like one that is about it.
    """
    q_terms = set(_tokens(query))
    if not q_terms:
        return 0.0
    counter = Counter(_tokens(document))
    if not counter:
        return 0.0
    covered = 0.0
    for term in q_terms:
        freq = counter.get(term, 0)
        if freq:
            covered += freq / (freq + 1.5)  # 1 hit → 0.40, 3 → 0.67, 10 → 0.87
    return covered / len(q_terms)


def bm25_score(query: str, document: str) -> float:
    """Single-document convenience wrapper. Prefer :func:`bm25_scores`."""
    return bm25_scores(query, [document])[0]
