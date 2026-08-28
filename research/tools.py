"""Research tool registration."""

from __future__ import annotations

import json

from research.fetch import fetch_page
from research.researcher import research_web_sync
from research.provider import search_with_fallback
from tools.registry import Tool, ToolRegistry


def register_research_tools(reg: ToolRegistry) -> None:
    reg.register(
        Tool(
            name="research_web",
            description=(
                "Research a question on the web: search, read pages, synthesize a direct "
                "answer with numbered citations and a Sources section. Use for factual or "
                "current-events questions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "depth": {"type": "string", "enum": ["quick", "thorough"]},
                },
                "required": ["question"],
            },
            handler=lambda question, depth="quick": json.dumps(
                research_web_sync(question, depth=depth), ensure_ascii=False
            ),
        )
    )

    async def _web_search(query: str, limit: int = 15) -> str:
        results = await search_with_fallback(query, limit=limit)
        return json.dumps(results, ensure_ascii=False)

    reg.register(
        Tool(
            name="web_search",
            description="Search the web; returns candidate URLs and snippets only.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
            handler=_web_search,
        )
    )

    async def _web_fetch(url: str) -> str:
        page = await fetch_page(url)
        return json.dumps(page, ensure_ascii=False)

    reg.register(
        Tool(
            name="web_fetch",
            description="Fetch and extract readable text from a single URL.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            handler=_web_fetch,
        )
    )
