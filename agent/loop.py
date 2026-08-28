"""Hermes-style ReAct agent loop."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from agent.prompt import build_system_prompt
from agent.tools_parse import has_tool_call_markup, parse_tool_call, strip_tool_call
from daemon.config import assistant_settings
from memory.store import get_store, read_memory_md, read_user_md
from models.catalog import get_runtime
from research.events import set_research_callback
from tools.builtin import register_builtin_tools
from tools.registry import get_registry

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict[str, Any]], None]

_TOOL_STATUS: dict[str, str] = {
    "research_web": "Searching the web…",
    "web_search": "Searching the web…",
    "web_fetch": "Reading page…",
    "shell": "Running command…",
    "memory_read": "Reading memory…",
    "memory_write": "Updating memory…",
    "session_search": "Searching past chats…",
    "skill_read": "Loading skill…",
    "skill_write": "Saving skill…",
    "schedule_task": "Scheduling task…",
    "run_background": "Starting background task…",
    "notify_user": "Sending notification…",
    "desktop_open": "Opening…",
    "todos_list": "Loading todos…",
    "todos_add": "Adding todo…",
    "todos_complete": "Updating todo…",
}

_tools_registered = False


def tool_status_message(name: str) -> str:
    if name.startswith("mcp."):
        return f"Calling {name.split('.')[-1]}…"
    return _TOOL_STATUS.get(name, f"Calling {name}…")


def _emit(on_event: EventCallback | None, kind: str, **payload: Any) -> None:
    if on_event:
        on_event(kind, payload)


def _stream_answer_tokens(text: str, on_event: EventCallback | None) -> None:
    if not on_event or not text:
        return
    for word in text.split():
        _emit(on_event, "token", text=word + " ")


def _ensure_tools() -> None:
    global _tools_registered
    if not _tools_registered:
        register_builtin_tools()
        _tools_registered = True


@dataclass
class AgentResult:
    answer: str
    session_id: str
    tool_calls: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)


class AIAgent:
    def __init__(self, session_id: str | None = None) -> None:
        _ensure_tools()
        self.store = get_store()
        self.session_id = session_id or self.store.new_session()
        settings = assistant_settings().get("assistant", {})
        self.iteration_budget = int(settings.get("iteration_budget", 12))

    def _system_prompt(self) -> str:
        return build_system_prompt(
            cached_user=read_user_md(),
            cached_memory=read_memory_md(),
        )

    async def run(
        self,
        user_message: str,
        *,
        on_event: EventCallback | None = None,
    ) -> AgentResult:
        self.store.add_message(self.session_id, "user", user_message)
        history = self.store.get_messages(self.session_id)
        runtime = get_runtime()
        tool_calls = 0
        final = ""

        def _research_cb(_kind: str, payload: dict[str, Any]) -> None:
            _emit(on_event, "research", message=payload.get("message", "Researching…"))

        set_research_callback(_research_cb)
        try:
            for _ in range(self.iteration_budget):
                messages = [{"role": "system", "content": self._system_prompt()}, *history]
                if not runtime.status.get("ready"):
                    final = (
                        "The NPU model is not loaded yet. Open Settings (gear icon) "
                        "and wait for the model warm-up to finish."
                    )
                    _emit(on_event, "status", message="Model not ready")
                    break

                _emit(on_event, "status", message="Thinking…")
                raw = runtime.chat(messages, max_new_tokens=768)
                call = parse_tool_call(raw)

                if not call and has_tool_call_markup(raw):
                    tool_calls += 1
                    _emit(on_event, "status", message="Parsing tool call…")
                    err = (
                        "Could not parse tool_call JSON. Use exactly:\n"
                        '<tool_call>\n{"name": "tool_name", "arguments": {"question": "..."}}\n</tool_call>'
                    )
                    history.append({"role": "assistant", "content": raw})
                    history.append(
                        {
                            "role": "user",
                            "content": f'<tool_result name="parse_error">\n{err}\n</tool_result>',
                        }
                    )
                    self.store.add_message(self.session_id, "assistant", raw)
                    self.store.add_message(self.session_id, "user", f"[tool:parse_error] {err}")
                    continue

                if call:
                    tool_calls += 1
                    name = call["name"]
                    args = call.get("arguments", {})
                    logger.info("tool call %s %s", name, args)
                    _emit(
                        on_event,
                        "tool",
                        name=name,
                        message=tool_status_message(name),
                    )

                    def _event_bridge(kind: str, payload: dict[str, Any]) -> None:
                        _emit(on_event, kind, **payload)

                    result = await get_registry().call(name, args, on_event=_event_bridge)
                    if name == "research_web":
                        try:
                            data = json.loads(result)
                            srcs = data.get("sources")
                            if srcs:
                                _emit(on_event, "sources", sources=srcs)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    history.append({"role": "assistant", "content": raw})
                    history.append(
                        {
                            "role": "user",
                            "content": f"<tool_result name=\"{name}\">\n{result}\n</tool_result>",
                        }
                    )
                    self.store.add_message(self.session_id, "assistant", raw)
                    self.store.add_message(self.session_id, "user", f"[tool:{name}] {result[:500]}")
                    continue

                final = strip_tool_call(raw) or raw.strip()
                self.store.add_message(self.session_id, "assistant", final)
                _stream_answer_tokens(final, on_event)
                _emit(on_event, "answer", text=final)
                break
            else:
                final = final or "I reached my iteration limit. Please try a simpler request."
                _stream_answer_tokens(final, on_event)
                _emit(on_event, "answer", text=final)
        finally:
            set_research_callback(None)

        await self._maybe_learn(user_message, tool_calls)
        return AgentResult(
            answer=final,
            session_id=self.session_id,
            tool_calls=tool_calls,
            messages=history,
        )

    async def run_stream(self, user_message: str) -> AsyncIterator[str]:
        result = await self.run(user_message)
        yield result.answer

    async def _maybe_learn(self, user_message: str, tool_calls: int) -> None:
        if tool_calls < 3:
            return
        runtime = get_runtime()
        if not runtime.status.get("ready"):
            return
        prompt = (
            "After this multi-step task, should a reusable skill be saved? "
            f"User asked: {user_message}\n"
            "If yes, call skill_write with agentskills.io frontmatter. "
            "If no, reply NO_SKILL."
        )
        try:
            resp = runtime.generate(prompt, max_new_tokens=256)
            if "NO_SKILL" in resp:
                return
            call = parse_tool_call(resp)
            if call and call["name"] == "skill_write":
                await get_registry().call(call["name"], call.get("arguments", {}))
        except Exception:  # noqa: BLE001
            logger.debug("learn loop skipped", exc_info=True)
