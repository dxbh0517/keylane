"""Hermes-style ReAct agent loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

from agent.prompt import assemble_for_turn
from agent.tools_parse import has_tool_call_markup, parse_tool_call
from daemon.config import assistant_settings
from memory.store import get_store
from seams import get_context
from npu.thinking import extract_user_answer, ran_out_mid_thought, sanitize_response
from research.events import set_research_callback
from seams.prompt import Assembly, latest_context_digest
from tools.goal_tools import register_goal_tools, render_goal
from tools.guards import reset_repeat_chain
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
    "remember": "Remembering that…",
    "recall": "Checking what I know…",
    "forget": "Forgetting that…",
    "memories_list": "Reviewing memory…",
    "session_search": "Searching past chats…",
    "skill_read": "Loading skill…",
    "skill_write": "Saving skill…",
    "schedule_task": "Scheduling task…",
    "remind_me": "Setting a reminder…",
    "reminders_list": "Checking reminders…",
    "reminder_cancel": "Cancelling reminder…",
    "watch_create": "Setting up a watcher…",
    "run_background": "Starting background task…",
    "inbox_list": "Checking your inbox…",
    "notify_user": "Sending notification…",
    "desktop_open": "Opening…",
    "todo_write": "Updating the plan…",
    "job_list": "Checking background work…",
    "job_output": "Collecting a result…",
    "job_kill": "Stopping a job…",
    "ask_user": "Waiting for you…",
    "skill_list": "Listing skills…",
}

# Tools whose result is an acknowledgement, not information. After one of
# these the model has nothing left to look up, so nudge it straight to a
# plain-text reply instead of letting it loop on another tool call.
_ACK_ONLY_TOOLS = frozenset({"remember", "forget", "memory_write", "remind_me", "reminder_cancel"})

# One bound, because there is now one transcript. Keeping a second, shorter copy
# for the database is what let the stored history drift from the history the
# model was shown within a turn.
_TOOL_RESULT_CHARS = 2800


def tool_status_message(name: str) -> str:
    if name.startswith("mcp."):
        return f"Calling {name.split('.')[-1]}…"
    return _TOOL_STATUS.get(name, f"Calling {name}…")


def _emit(on_event: EventCallback | None, kind: str, **payload: Any) -> None:
    if on_event:
        on_event(kind, payload)


def _clean_for_user(text: str) -> str:
    return extract_user_answer(text)


def _emit_user_answer(on_event: EventCallback | None, text: str) -> str:
    clean = _clean_for_user(text)
    if on_event:
        _emit(on_event, "replace_answer", text=clean)
    _emit(on_event, "answer", text=clean)
    return clean


def _normalize_research_question(question: str) -> str:
    q = question.lower().strip()
    q = re.sub(r"[^\w\s]", " ", q)
    return " ".join(q.split())


def _tool_signature(name: str, args: dict[str, Any]) -> str:
    if name == "research_web":
        q = _normalize_research_question(str(args.get("question", "")))
        return json.dumps({"name": name, "question": q}, sort_keys=True)
    return json.dumps({"name": name, "arguments": args}, sort_keys=True, ensure_ascii=False)


def _assistant_history_stub(call: dict[str, Any]) -> str:
    return json.dumps(
        {"tool_call": call["name"], "arguments": call.get("arguments", {})},
        ensure_ascii=False,
    )


def _compress_tool_result(name: str, result: str) -> str:
    """Shrink tool payloads for model context while keeping answers usable."""
    if name == "research_web":
        try:
            data = json.loads(result)
            answer = str(data.get("answer", "")).strip()
            sources = data.get("sources") or []
            if answer:
                return json.dumps(
                    {
                        "answer": answer,
                        "sources": sources[:8],
                        "note": "Reply with the answer field. Do not call research_web again.",
                    },
                    ensure_ascii=False,
                )
        except (json.JSONDecodeError, TypeError):
            pass
    if len(result) > _TOOL_RESULT_CHARS:
        return result[:_TOOL_RESULT_CHARS] + "…"
    return result


def tool_result_block(name: str, body: str, *, note: str = "") -> str:
    """The one rendering of a tool result — the model sees exactly this string.

    It is also exactly what is written to the session store, so replaying a
    session reconstructs the request the model actually answered.
    """
    payload = f"{body}\n{note}" if note else body
    return f'<tool_result name="{name}">\n{payload}\n</tool_result>'


def _ensure_tools() -> None:
    """Compose the context, which registers the tools and the prompt sections.

    Both used to happen here directly; going through the context means they
    happen exactly once however the agent is reached — a turn, a scheduled job,
    or a background run.
    """
    from seams import get_context

    get_context()


@dataclass
class AgentResult:
    answer: str
    session_id: str
    tool_calls: int = 0
    messages: list[dict[str, str]] = field(default_factory=list)


class AIAgent:
    def __init__(
        self,
        session_id: str | None = None,
        *,
        tools: Any = None,
        route: str = "interactive",
    ) -> None:
        _ensure_tools()
        self.store = get_store()
        self.session_id = session_id or self.store.new_session()
        # A scope of its own, so spilled results land under this session and a
        # subagent can be handed a restricted child of it.
        self.tools = tools if tools is not None else get_registry().child(self.session_id)
        # Goal tools are bound to one conversation, so they register into this
        # session's own scope rather than globally.
        register_goal_tools(self.tools, self.session_id)
        # A purpose, not a model. `interactive` keeps the HUD on the NPU;
        # delegated and scheduled work asks for `background`, which prefers the
        # larger model when one is configured.
        self.route = route
        settings = assistant_settings().get("assistant", {})
        self.iteration_budget = int(settings.get("iteration_budget", 12))

    def _assemble(self) -> Assembly:
        """One assembly for the whole turn.

        Freezing it here keeps the system message byte-identical across every
        step of a turn — the clock inside the dynamic block would otherwise tick
        mid-turn and append a second copy for no reason.
        """
        # The active pipeline's real limit, so a prompt that cannot fit sheds
        # optional guidance instead of being truncated mid-sentence — which on
        # the NPU would cut the tool-call format off the end.
        budget = get_context().llm.prompt_budget_chars(self.route)
        return assemble_for_turn(
            extra_contexts=[render_goal(self.session_id)],
            budget_chars=budget,
        )

    def _invoked_skills(self, user_message: str) -> list[str]:
        """Rendered bodies for every skill the user named with `/name`."""
        from seams import get_context
        from seams.skills import find_invocations, render_skill

        registry = get_context().skills
        blocks: list[str] = []
        for name in find_invocations(user_message, registry.list()):
            skill = registry.get(name)
            if skill is not None:
                blocks.append(render_skill(skill))
        return blocks

    def _publish_sources(
        self,
        result: str,
        *,
        on_event: EventCallback | None,
    ) -> str:
        """Send the sources to the HUD and keep the answer as a fallback.

        The research answer is no longer emitted as the reply — the model gets
        it as a tool result and writes the reply itself. But the sources belong
        on the card either way, and the answer is worth holding onto in case the
        turn runs out of iterations before the model says anything.
        """
        try:
            data = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return ""

        srcs = data.get("sources")
        if srcs:
            _emit(on_event, "sources", sources=srcs)

        answer = str(data.get("answer") or "").strip()
        return sanitize_response(answer) if answer else ""

    async def _generate(
        self,
        messages: list[dict[str, str]],
        llm: Any,
        images: list[bytes] | None,
        on_event: EventCallback | None,
    ) -> str:
        """One model call, off the event loop, streaming as it goes.

        Two things were wrong here and they compounded. The call is
        synchronous and took tens of seconds, and it was made directly inside a
        coroutine — so for its whole duration the daemon's event loop could not
        run and no SSE event already queued could leave the process. And it
        passed ``on_token=None``, so there was nothing to send anyway, on a
        device where the wait is long enough that watching text appear is the
        difference between usable and not.

        Tokens are handed back through ``loop.call_soon_threadsafe`` because
        the runtime calls the streamer from its own thread, and the queue on
        the other end of ``on_event`` is not thread-safe.
        """
        loop = asyncio.get_running_loop()

        forward: Callable[[str], None] | None = None
        if on_event is not None:
            # Only a plain answer is streamed. A tool call arrives as markup the
            # user must never see, so the first sign of one stops the stream and
            # the HUD keeps whatever status it had.
            state = {"suppressed": False, "seen": ""}

            def _post(kind: str, text: str) -> None:
                # call_soon_threadsafe takes positional arguments only, so the
                # payload is closed over rather than passed as keywords.
                loop.call_soon_threadsafe(lambda: _emit(on_event, kind, text=text))

            def _on_token(piece: str) -> None:
                if state["suppressed"]:
                    return
                state["seen"] += piece
                if has_tool_call_markup(state["seen"]) or "<tool_call" in state["seen"]:
                    state["suppressed"] = True
                    _post("replace_answer", "")
                    return
                _post("token", piece)

            forward = _on_token

        return await asyncio.to_thread(
            llm.chat,
            messages,
            route=self.route,
            max_new_tokens=512,
            images=images,
            on_token=forward,
        )

    async def run(
        self,
        user_message: str,
        *,
        on_event: EventCallback | None = None,
        images: list[bytes] | None = None,
    ) -> AgentResult:
        if images:
            note = " [image attached]" if user_message else "[image attached]"
            self.store.add_message(self.session_id, "user", f"{user_message}{note}".strip())
        else:
            self.store.add_message(self.session_id, "user", user_message)
        history = self.store.get_messages(self.session_id)
        # A new instruction is not a loop, whatever the last turn was doing.
        reset_repeat_chain(self.session_id)

        def _record(role: str, content: str) -> None:
            """Append to the model's history and the session log as one act.

            There is exactly one transcript. Writing a different string to the
            store than the one placed in `history` is what previously let the
            model read `[tool:x] …` on the next turn after being shown
            `<tool_result name="x">` on this one.
            """
            history.append({"role": role, "content": content})
            self.store.add_message(self.session_id, role, content)

        # A `/name` the user typed injects that skill's body directly. It is the
        # only way to reach a skill the model may not load itself, and it costs
        # the tokens deterministically, at the user's request, rather than at
        # the model's discretion.
        for block in self._invoked_skills(user_message):
            _record("user", block)

        llm = get_context().llm
        tool_calls = 0
        final = ""
        last_research_answer = ""
        last_tool_signature = ""
        research_cache: dict[str, str] = {}

        def _research_cb(_kind: str, payload: dict[str, Any]) -> None:
            _emit(on_event, "research", message=payload.get("message", "Researching…"))

        set_research_callback(_research_cb)
        try:
            assembly = self._assemble()
            for _ in range(self.iteration_budget):
                # The dynamic block is appended to history, never folded into
                # the system message, and only when what it says has changed
                # since the newest copy the model can still see.
                if assembly.context and assembly.context_digest != latest_context_digest(history):
                    _record("user", assembly.context)
                messages = [{"role": "system", "content": assembly.system}, *history]
                if not llm.is_ready(self.route):
                    final = _emit_user_answer(
                        on_event,
                        "No model is ready yet. Open Settings (gear icon) and wait for "
                        "the model warm-up to finish.",
                    )
                    _emit(on_event, "status", message="Model not ready")
                    break

                _emit(on_event, "status", message="Thinking…")
                if on_event:
                    _emit(on_event, "replace_answer", text="")

                raw = await self._generate(messages, llm, images, on_event)
                call = parse_tool_call(raw)

                if not call and has_tool_call_markup(raw):
                    tool_calls += 1
                    _emit(on_event, "status", message="Parsing tool call…")
                    err = (
                        "Could not parse tool_call JSON. Use exactly:\n"
                        '<tool_call>\n{"name": "tool_name", "arguments": {"question": "..."}}\n</tool_call>'
                    )
                    stub = _clean_for_user(raw) or "[unparsed tool_call]"
                    _record("assistant", stub)
                    _record("user", tool_result_block("parse_error", err))
                    continue

                if call:
                    tool_calls += 1
                    name = call["name"]
                    args = call.get("arguments", {})
                    signature = _tool_signature(name, args)

                    if signature == last_tool_signature:
                        err = (
                            "You already called this tool with the same arguments. "
                            "Use the prior tool_result and reply to the user in plain text. "
                            "Do not emit another tool_call block."
                        )
                        _record("assistant", _assistant_history_stub(call))
                        _record("user", tool_result_block("duplicate_call", err))
                        continue

                    last_tool_signature = signature
                    logger.info("tool call %s %s", name, args)
                    _emit(
                        on_event,
                        "tool",
                        name=name,
                        message=tool_status_message(name),
                    )

                    def _event_bridge(kind: str, payload: dict[str, Any]) -> None:
                        _emit(on_event, kind, **payload)

                    if name == "research_web":
                        # A normal tool result, not the end of the turn. Ending
                        # here is what stopped Keylane ever researching a thing
                        # *and* remembering it, or researching two things.
                        cache_key = _normalize_research_question(str(args.get("question", "")))
                        result = research_cache.get(cache_key)
                        if result is None:
                            result = await self.tools.call(
                                name, args, on_event=_event_bridge
                            )
                            research_cache[cache_key] = result
                        last_research_answer = self._publish_sources(result, on_event=on_event)
                        _record("assistant", _assistant_history_stub(call))
                        _record("user", tool_result_block(name, _compress_tool_result(name, result)))
                        continue

                    if name in _ACK_ONLY_TOOLS:
                        ack = await self.tools.call(name, args, on_event=_event_bridge)
                        remainder = _clean_for_user(raw)
                        if remainder:
                            final = _emit_user_answer(on_event, remainder)
                            _record("assistant", final)
                            break
                        _record("assistant", _assistant_history_stub(call))
                        _record(
                            "user",
                            tool_result_block(
                                name,
                                _compress_tool_result(name, ack),
                                note="Done. Now confirm this to the user in one short plain-text sentence.",
                            ),
                        )
                        continue

                    result = await self.tools.call(name, args, on_event=_event_bridge)
                    _record("assistant", _assistant_history_stub(call))
                    _record("user", tool_result_block(name, _compress_tool_result(name, result)))
                    continue

                final = _clean_for_user(raw)
                if not final:
                    if has_tool_call_markup(raw):
                        final = _emit_user_answer(
                            on_event,
                            "I could not complete that request. "
                            "Try asking again or check that web search (SearXNG) is running.",
                        )
                    elif ran_out_mid_thought(raw):
                        # The model reasoned right up to the token limit and
                        # never reached an answer. Saying "I could not produce
                        # a response" is untrue and unactionable: it produced
                        # plenty, and the fix is a bigger budget or a model
                        # that does not think as hard.
                        logger.warning(
                            "model used all %d tokens reasoning without answering", 512
                        )
                        final = _emit_user_answer(
                            on_event,
                            "The model spent its whole reply thinking and never got to "
                            "an answer. Try a shorter question, or pick a model that "
                            "reasons less in Settings → Model.",
                        )
                    else:
                        final = _emit_user_answer(
                            on_event,
                            "I could not produce a response. Please try again.",
                        )
                else:
                    final = _emit_user_answer(on_event, final)
                _record("assistant", final)
                break
            else:
                fallback = last_research_answer or (
                    "I reached my iteration limit. Please try a simpler request."
                )
                final = _emit_user_answer(on_event, fallback)
                if last_research_answer:
                    _record("assistant", final)
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
        """Optionally propose a reusable skill after a multi-step task.

        Off by default. Writing a file that changes how the assistant behaves
        later is not something to do on the model's own initiative, and the
        extra generation it costs lands on exactly the turns that were already
        the slowest. With it on, `skill_write` still goes through the
        permission gate.
        """
        settings = assistant_settings().get("assistant", {})
        if not settings.get("auto_learn_skills", False):
            return
        if tool_calls < 3:
            return
        llm = get_context().llm
        if not llm.is_ready("utility"):
            return
        prompt = (
            "After this multi-step task, should a reusable skill be saved? "
            f"User asked: {user_message}\n"
            "If yes, call skill_write with agentskills.io frontmatter. "
            "If no, reply NO_SKILL."
        )
        try:
            resp = llm.generate(prompt, route="utility", max_new_tokens=256)
            if "NO_SKILL" in resp:
                return
            call = parse_tool_call(resp)
            if call and call["name"] == "skill_write":
                await self.tools.call(call["name"], call.get("arguments", {}))
        except Exception:  # noqa: BLE001
            logger.debug("learn loop skipped", exc_info=True)
