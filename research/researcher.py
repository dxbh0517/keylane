"""Agentic web research — plan, search, read, compress, synthesize.

Sources travel beside the answer rather than inside it; the interface renders
attribution, so nothing here writes a Sources section or a [1] marker.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

from daemon.config import research_settings
from seams import get_context
from npu.thinking import sanitize_response
from research.events import emit_research
from research.provider import (
    bm25_scores,
    coverage_score,
    get_extract_provider,
    search_with_fallback,
)
from research.search import diversify_candidates

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], None]

_CHUNK_CHARS = 600
_MAX_CHUNKS = 8


@dataclass
class Source:
    index: int
    title: str
    url: str
    excerpt: str = ""


@dataclass
class ResearchResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


def _emit(message: str, **payload: Any) -> None:
    emit_research("research", message=message, **payload)


def _plan_queries_heuristic(question: str, *, max_queries: int) -> list[str]:
    q = question.strip()
    queries = [q]
    words = q.lower().split()
    if len(words) > 6:
        queries.append(" ".join(words[:8]))
    if "latest" in q.lower() or "2026" in q or "2025" in q:
        queries.append(f"{q} news update")
    return list(dict.fromkeys(queries))[:max_queries]


def _plan_queries(question: str, *, max_queries: int) -> list[str]:
    llm = get_context().llm
    if not llm.is_ready("utility"):
        return _plan_queries_heuristic(question, max_queries=max_queries)

    _emit("Planning search queries…")
    prompt = (
        f'Expand this question into up to {max_queries} web search queries.\n'
        f'Reply with ONLY JSON: {{"queries": ["..."], "intent": "news|fact|howto"}}\n\n'
        f"Question: {question}"
    )
    try:
        raw = llm.generate(prompt, route="utility", max_new_tokens=128)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            queries = data.get("queries", [])
            if isinstance(queries, list) and queries:
                return [str(q).strip() for q in queries if str(q).strip()][:max_queries]
    except Exception as exc:  # noqa: BLE001
        logger.debug("model query planning failed: %s", exc)

    return _plan_queries_heuristic(question, max_queries=max_queries)


def _score_relevance(question: str, text: str) -> float:
    """Absolute relevance in [0, 1] — safe to compare against a threshold."""
    return coverage_score(question, text)


def _prerank_candidates(question: str, candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Rank candidates against each other; the candidate set is the corpus."""
    if not candidates:
        return []
    blobs = [f"{c.get('title', '')} {c.get('snippet', '')}" for c in candidates]
    scored = list(zip(bm25_scores(question, blobs), candidates))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored]


def _select_urls(
    question: str,
    candidates: list[dict[str, str]],
    max_urls: int,
) -> list[dict[str, str]]:
    if not candidates:
        return []

    ranked = _prerank_candidates(question, candidates)
    diversified = diversify_candidates(ranked, max_urls * 2)
    llm = get_context().llm
    if not llm.is_ready("utility"):
        return diversify_candidates(ranked, max_urls)

    listing = "\n".join(
        f"{i+1}. [{c['title']}] {c['url']}"
        for i, c in enumerate(diversified[:15])
    )
    prompt = (
        f"Question: {question}\n\n"
        f"For each URL below reply yes/no if it likely helps answer the question.\n"
        f"Format: one line per number, e.g. '1 yes\\n2 no'\n\n{listing}"
    )
    try:
        raw = llm.generate(prompt, route="utility", max_new_tokens=128)
        picked: list[dict[str, str]] = []
        for line in raw.splitlines():
            m = re.match(r"\s*(\d+)\s*(yes|y|true|1)", line, re.IGNORECASE)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(diversified):
                picked.append(diversified[idx])
            if len(picked) >= max_urls:
                break
        if picked:
            return picked
    except Exception as exc:  # noqa: BLE001
        logger.debug("model url selection failed: %s", exc)

    return diversify_candidates(ranked, max_urls)


def _chunk_text(text: str, chunk_size: int = _CHUNK_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            if len(para) <= chunk_size:
                current = para
            else:
                for i in range(0, len(para), chunk_size):
                    chunks.append(para[i : i + chunk_size])
                current = ""
    if current:
        chunks.append(current)
    return chunks


def _compress_evidence(
    question: str,
    pages: list[dict[str, str]],
    sources: list[Source],
) -> str:
    chunks: list[tuple[str, Source]] = []
    for page, source in zip(pages, sources):
        for chunk in _chunk_text(page.get("text", "")):
            chunks.append((chunk, source))
    if not chunks:
        return ""

    # Every chunk from every page is one corpus, so a phrase that appears in
    # one chunk outweighs one repeated across all of them.
    ranked = bm25_scores(question, [text for text, _ in chunks])
    scored_chunks = [
        (score, source.index, text, source)
        for score, (text, source) in zip(ranked, chunks)
    ]
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    top = scored_chunks[:_MAX_CHUNKS]

    parts: list[str] = []
    for score, idx, chunk, source in top:
        if score <= 0:
            continue
        parts.append(f"[{idx}] {source.title} ({source.url})\n{chunk}")

    return "\n\n".join(parts)


def _synthesize(
    question: str,
    pages: list[dict[str, str]],
    sources: list[Source],
) -> str:
    llm = get_context().llm
    evidence = _compress_evidence(question, pages, sources)

    if llm.is_ready("background") and evidence:
        _emit("Synthesizing answer…")
        # Attribution is the interface's job: the HUD renders the source list
        # from the `sources` event, and ui/canvas.py strips any [1] markers and
        # Sources heading the model writes anyway. Asking for citations here
        # only spent tokens the NPU budget cannot spare, and put this prompt in
        # direct conflict with the system prompt's output contract.
        prompt = (
            "You are a research assistant. Answer the user's question directly using ONLY "
            "the evidence below. Write the answer itself — no citation markers, no Sources "
            "section, no preamble.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence[:2800]}\n\nAnswer:"
        )
        try:
            return sanitize_response(
                llm.generate(prompt, route="background", max_new_tokens=800).strip()
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("synthesis failed: %s", exc)

    # Fallback when no model is loaded: the excerpts themselves, unattributed
    # in the text because the sources travel beside the answer.
    lines: list[str] = []
    for page in pages:
        snippet = page["text"][:400].replace("\n", " ").strip()
        if snippet:
            lines.append(f"{snippet}…")
    return sanitize_response("\n\n".join(lines))


def _domain_label(url: str) -> str:
    host = (urlparse(url).hostname or url).lower()
    if host.startswith("www."):
        host = host[4:]
    return host


async def research_web(
    question: str,
    *,
    depth: str = "quick",
) -> ResearchResult:
    cfg = research_settings()
    max_queries = int(cfg.get("max_search_queries", 3))
    max_urls = int(cfg.get("max_urls_per_round", 8))
    max_rounds = int(cfg.get("max_fetch_rounds", 2))
    if depth == "thorough":
        max_urls = min(max_urls + 4, 12)
        max_rounds = min(max_rounds + 1, 3)

    trace: list[str] = []
    all_candidates: list[dict[str, str]] = []

    queries = _plan_queries(question, max_queries=max_queries)
    trace.append(f"planned queries: {queries}")

    extract = get_extract_provider()
    for i, q in enumerate(queries, 1):
        _emit(f"Searching the web ({i}/{len(queries)})…", query=q)
        found = await search_with_fallback(q, limit=int(cfg.get("max_candidates_per_query", 20)))
        trace.append(f"search '{q}': {len(found)} results")
        all_candidates.extend(found)

    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for c in all_candidates:
        u = c["url"]
        if u in seen:
            continue
        seen.add(u)
        unique.append(c)

    pages: list[dict[str, str]] = []
    sources: list[Source] = []

    for round_num in range(max_rounds):
        remaining = max_urls - len(pages)
        if remaining <= 0:
            break
        to_fetch = _select_urls(question, unique, remaining)
        trace.append(f"round {round_num+1}: fetching {len(to_fetch)} urls")

        for cand in to_fetch:
            label = _domain_label(cand["url"])
            _emit(f"Reading {label}…", url=cand["url"])
            try:
                page = await extract.extract(cand["url"])
            except Exception as exc:  # noqa: BLE001
                trace.append(f"fetch failed {cand['url']}: {exc}")
                continue
            status = int(page.get("status_code") or 200)
            if status >= 400:
                # A 404 body is an error page; its text would pollute the evidence.
                trace.append(f"skipped HTTP {status} {cand['url']}")
                continue
            rel = _score_relevance(question, page["text"])
            if rel < 0.05 and len(pages) >= 2:
                trace.append(f"skipped low relevance {cand['url']} ({rel:.2f})")
                continue
            idx = len(sources) + 1
            sources.append(
                Source(
                    index=idx,
                    title=page.get("title") or cand["title"],
                    url=cand["url"],
                    excerpt=page["text"][:300],
                )
            )
            pages.append(page)
            unique = [c for c in unique if c["url"] != cand["url"]]

        if pages and round_num == 0 and _score_relevance(question, pages[-1]["text"]) > 0.4:
            break

    if not pages:
        return ResearchResult(
            answer=(
                "I could not find usable web sources for that question. "
                "Check Settings → Web and ensure a search provider is reachable."
            ),
            sources=[],
            trace=trace,
        )

    # No Sources section is appended: `sources` rides alongside the answer and
    # the interface renders it.
    answer = _synthesize(question, pages, sources)
    return ResearchResult(answer=answer, sources=sources, trace=trace)


def research_web_sync(question: str, depth: str = "quick") -> dict[str, Any]:
    """Sync wrapper — safe to call from threads; not from a running event loop."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _run() -> ResearchResult:
        return asyncio.run(research_web(question, depth=depth))

    try:
        asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(_run).result()
    except RuntimeError:
        result = _run()

    return {
        "answer": result.answer,
        "sources": [
            {"index": s.index, "title": s.title, "url": s.url} for s in result.sources
        ],
        "trace": result.trace,
    }
