"""One whole turn, end to end, against a scripted model.

These are the tests that would have caught the transcript drift: they assert on
what the *store* holds after the turn, which is what the next turn reads.
"""

from __future__ import annotations

import asyncio
import json

import pytest


@pytest.fixture()
def agent(tmp_path, monkeypatch):
    """An agent on a throwaway database with a scripted model."""
    from memory.store import SessionStore
    from seams import build_context, context as context_module
    from seams.prompt import SystemPrompt

    store = SessionStore(tmp_path / "turn.db")
    monkeypatch.setattr("agent.loop.get_store", lambda: store)
    monkeypatch.setattr("seams.goals.get_store", lambda: store)

    ctx = build_context()
    monkeypatch.setattr(context_module, "_context", ctx)

    replies: list[str] = []

    class ScriptedAdapter:
        id = "npu"

        def available(self) -> bool:
            return True

        @property
        def status(self):
            return {"kind": "scripted"}

        def generate(self, prompt, **kwargs):
            return replies.pop(0) if replies else "NO_SKILL"

        def chat(self, messages, **kwargs):
            self.last_messages = messages
            return replies.pop(0) if replies else "Done."

    adapter = ScriptedAdapter()
    for existing in list(ctx.llm.adapters()):
        if existing.id == "npu":
            ctx.llm._adapters.pop("npu")  # noqa: SLF001
    ctx.llm.register(adapter)

    from agent.loop import AIAgent

    built = AIAgent()
    return built, replies, adapter, store


def _run(agent_tuple, message: str):
    built, *_ = agent_tuple
    return asyncio.run(built.run(message))


def test_a_plain_answer_reaches_the_user(agent) -> None:
    built, replies, _, _ = agent
    replies.append("Fedora 44 is current.")
    assert _run(agent, "which fedora is current?").answer == "Fedora 44 is current."


def test_the_system_prompt_is_identical_across_turns(agent) -> None:
    """A prompt that changes every turn can never reuse a cached prefix."""
    built, replies, adapter, _ = agent
    replies.extend(["First.", "Second."])
    _run(agent, "one")
    first = adapter.last_messages[0]["content"]
    _run(agent, "two")
    assert adapter.last_messages[0]["content"] == first


def test_volatile_facts_ride_in_the_conversation_not_the_system_message(agent) -> None:
    built, replies, adapter, _ = agent
    replies.append("Done.")
    _run(agent, "hello")
    system = adapter.last_messages[0]["content"]
    rest = "\n".join(m["content"] for m in adapter.last_messages[1:])
    assert "Local time:" not in system
    assert "<session_context>" in rest


def test_the_context_block_is_not_repeated_within_a_turn(agent) -> None:
    """It only re-emits when what it says has changed."""
    built, replies, _, store = agent
    replies.extend(
        [
            '<tool_call>\n{"name": "todo_write", "arguments": {"todos": [{"content": "a", "status": "pending"}]}}\n</tool_call>',
            "Done.",
        ]
    )
    _run(agent, "plan something")
    blocks = [
        m for m in store.get_messages(built.session_id) if "<session_context>" in m["content"]
    ]
    assert len(blocks) == 1


def test_the_stored_transcript_is_what_the_model_was_shown(agent) -> None:
    """The bug this replaced wrote `[tool:x]` to the store and XML to history."""
    built, replies, adapter, store = agent
    replies.extend(
        [
            '<tool_call>\n{"name": "todo_write", "arguments": {"todos": []}}\n</tool_call>',
            "Cleared the list.",
        ]
    )
    _run(agent, "clear my todos")

    stored = store.get_messages(built.session_id)
    shown = [m["content"] for m in adapter.last_messages[1:]]
    tool_results = [m["content"] for m in stored if m["content"].startswith("<tool_result")]
    assert tool_results, "no tool result was recorded"
    for block in tool_results:
        assert block in shown
        assert "[tool:" not in block


def test_a_tool_result_is_a_normal_step_not_the_end_of_the_turn(agent) -> None:
    """research_web used to end the turn, so it could never be combined."""
    built, replies, _, store = agent
    replies.extend(
        [
            '<tool_call>\n{"name": "todo_write", "arguments": {"todos": [{"content": "x", "status": "pending"}]}}\n</tool_call>',
            "Added it.",
        ]
    )
    result = _run(agent, "add x to my list")
    assert result.answer == "Added it."
    assert result.tool_calls == 1


def test_an_identical_repeat_is_answered_rather_than_repeated(agent) -> None:
    built, replies, _, _ = agent
    call = '<tool_call>\n{"name": "todo_write", "arguments": {"todos": []}}\n</tool_call>'
    replies.extend([call, call, "Nothing left."])
    assert _run(agent, "clear it").answer == "Nothing left."


def test_an_unparseable_tool_call_gets_the_format_back(agent) -> None:
    built, replies, _, store = agent
    replies.extend(["<tool_call>\nnot json at all\n</tool_call>", "Sorry, done now."])
    _run(agent, "do something")
    stored = "\n".join(m["content"] for m in store.get_messages(built.session_id))
    assert "parse_error" in stored


def test_an_unknown_tool_is_reported_with_what_exists(agent) -> None:
    built, replies, _, store = agent
    replies.extend(
        ['<tool_call>\n{"name": "nonesuch", "arguments": {}}\n</tool_call>', "Cannot do that."]
    )
    _run(agent, "use the nonesuch tool")
    stored = "\n".join(m["content"] for m in store.get_messages(built.session_id))
    assert "UNKNOWN_TOOL" in stored


def test_the_next_turn_reads_the_previous_turns_transcript(agent) -> None:
    built, replies, adapter, _ = agent
    replies.extend(["First answer.", "Second answer."])
    _run(agent, "first question")
    _run(agent, "second question")
    conversation = "\n".join(m["content"] for m in adapter.last_messages)
    assert "first question" in conversation
    assert "First answer." in conversation


def test_no_model_is_a_clear_message_not_a_crash(agent, monkeypatch) -> None:
    built, replies, adapter, _ = agent
    monkeypatch.setattr(adapter, "available", lambda: False)
    assert "No model is ready" in _run(agent, "hello").answer


# ── streaming ────────────────────────────────────────────────────────────


def test_a_tool_call_is_never_streamed_to_the_user():
    """Tokens go to the HUD as they arrive, and tool markup must not.

    The model emits a tool call as text, one token at a time, exactly like an
    answer. Streaming it would show the user a JSON block before the turn had
    done anything. The first sign of markup stops the stream and clears what
    was shown.
    """
    import asyncio

    from agent.loop import AIAgent

    events: list[tuple[str, dict]] = []

    class _Llm:
        def chat(self, messages, *, route, max_new_tokens, images, on_token):
            for piece in ('<tool', '_call>\n{"name": "recall"', "}\n</tool_call>"):
                if on_token:
                    on_token(piece)
            return '<tool_call>\n{"name": "recall"}\n</tool_call>'

    async def _run() -> None:
        agent = AIAgent.__new__(AIAgent)
        agent.route = "interactive"
        await agent._generate([], _Llm(), None, lambda k, p: events.append((k, p)))

    asyncio.run(_run())

    kinds = [k for k, _ in events]
    assert "replace_answer" in kinds, "the stream should have been cleared"
    streamed = "".join(p.get("text", "") for k, p in events if k == "token")
    assert "recall" not in streamed
    assert "tool_call" not in streamed


def test_a_plain_answer_streams_through():
    import asyncio

    from agent.loop import AIAgent

    events: list[tuple[str, dict]] = []

    class _Llm:
        def chat(self, messages, *, route, max_new_tokens, images, on_token):
            for piece in ("Paris ", "is ", "the ", "capital."):
                if on_token:
                    on_token(piece)
            return "Paris is the capital."

    async def _run() -> None:
        agent = AIAgent.__new__(AIAgent)
        agent.route = "interactive"
        return await agent._generate([], _Llm(), None, lambda k, p: events.append((k, p)))

    answer = asyncio.run(_run())

    streamed = "".join(p.get("text", "") for k, p in events if k == "token")
    assert streamed == "Paris is the capital."
    assert answer == "Paris is the capital."


def test_generation_does_not_block_the_event_loop():
    """The call is synchronous and long; it has to run off the loop.

    While it ran inline, no SSE event queued during a generation could leave
    the process — which is why streaming delivered nothing even once it was
    switched on.
    """
    import asyncio
    import threading

    from agent.loop import AIAgent

    generating = threading.Event()
    release = threading.Event()

    class _Llm:
        def chat(self, messages, *, route, max_new_tokens, images, on_token):
            generating.set()
            release.wait(5)
            return "done"

    async def _run() -> str:
        agent = AIAgent.__new__(AIAgent)
        agent.route = "interactive"
        task = asyncio.ensure_future(agent._generate([], _Llm(), None, None))
        # If the generate call held the loop, this would never get to run.
        while not generating.is_set():
            await asyncio.sleep(0.01)
        release.set()
        return await task

    assert asyncio.run(_run()) == "done"
