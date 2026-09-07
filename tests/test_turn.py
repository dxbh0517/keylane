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


# ── the runtime must not clean what the agent has to read ────────────────
#
# `RuntimeState.generate` used to end with `return sanitize_response(raw)`.
# That strips `<tool_call>` blocks along with `<think>` ones, so the agent —
# the one caller that has to *read* a tool call — received the single string
# that could never contain one. Every local tool call, MCP servers included,
# was destroyed between the model emitting it and the loop parsing it.
#
# It survived the suite because every test here stubs the LLM seam, so none of
# them ever ran the sanitizing that the real runtime did.


def test_sanitizing_destroys_a_tool_call() -> None:
    """The premise. If this ever stops being true, the guard below is moot."""
    from npu.thinking import sanitize_response

    from agent.tools_parse import parse_tool_call

    raw = (
        "<think>\nI should look at their mail.\n</think>\n\n"
        '<tool_call>\n{"name": "mcp.mailspring.list_accounts", "arguments": {}}\n</tool_call>'
    )
    assert parse_tool_call(raw) is not None
    assert parse_tool_call(sanitize_response(raw)) is None


def test_the_runtime_hands_back_what_the_model_decoded() -> None:
    """Cleaning is the consumer's job: each one wants something different.

    The agent parses markup, the HUD hides it, an OpenAI client wants prose.
    """
    import inspect

    from models.catalog import LocalModelRuntime

    source = inspect.getsource(LocalModelRuntime.generate)
    assert "return sanitize_response(raw)" not in source
    assert source.rstrip().endswith("return raw")


def test_the_openai_route_still_returns_prose() -> None:
    """It serves clients that want an answer, not this model's reasoning."""
    import inspect

    from daemon import openai_api

    source = inspect.getsource(openai_api.chat_completions)
    assert "extract_user_answer(raw)" in source


# ── a tool called with an invented argument ──────────────────────────────
#
# jan-nano called mcp.mailspring.list_unread_threads with
# {'accountId': 'user_account_id'} — a value it made up rather than reading
# from list_accounts. The call failed, and the model reported that to the user
# as "I don't have access to your email threads", which is the one conclusion
# that is not true: it has the tool and called it wrong.


@pytest.mark.parametrize(
    "value",
    ["user_account_id", "<ACCOUNT-ID>", "your email", "example", "TBD", "your_account_id"],
)
def test_invented_values_are_recognised(value: str) -> None:
    from agent.loop import looks_like_a_placeholder

    assert looks_like_a_placeholder(value)


@pytest.mark.parametrize("value", ["73b9dacc", "Inbox", "omar@example.com", "2026-09-07"])
def test_real_values_are_left_alone(value: str) -> None:
    """A false positive would nag a model that did everything right."""
    from agent.loop import looks_like_a_placeholder

    assert not looks_like_a_placeholder(value)


def test_non_strings_are_not_placeholders() -> None:
    from agent.loop import looks_like_a_placeholder

    assert not looks_like_a_placeholder(7)
    assert not looks_like_a_placeholder(None)
    assert not looks_like_a_placeholder({"nested": "id"})


def test_the_note_refuses_the_conclusion_the_model_reached() -> None:
    from agent.loop import tool_failure_note

    note = tool_failure_note(
        "mcp.mailspring.list_unread_threads",
        {"accountId": "user_account_id"},
        # The tool being called has to be in the list, or this is the other
        # case entirely — an invented name, which gets opposite advice.
        ["mcp.mailspring.list_unread_threads", "mcp.mailspring.list_accounts", "shell"],
    )
    assert "not a missing capability" in note
    assert "do not tell the user you lack access" in note
    assert "`accountId`" in note
    # It names the sibling that would produce the real value.
    assert "mcp.mailspring.list_accounts" in note


def test_a_failure_with_sound_arguments_gets_different_advice() -> None:
    """Nothing was guessed, so telling it to go and look would be wrong."""
    from agent.loop import tool_failure_note

    note = tool_failure_note("shell", {"command": "ls"}, ["shell"])
    assert "placeholder" not in note
    assert "schema" in note


@pytest.mark.parametrize(
    ("result", "is_error"),
    [
        ('{"error": "no such account", "code": "MCP_ERROR"}', True),
        ("Error: the Keylane UI is not running", True),
        ('{"threads": [], "count": 0}', False),
        ("You have 3 unread threads.", False),
        ("", False),
    ],
)
def test_only_failures_are_steered(result: str, is_error: bool) -> None:
    from agent.loop import _looks_like_an_error

    assert _looks_like_an_error(result) is is_error


# ── a tool name the model invented ───────────────────────────────────────
#
# The first version of tool_failure_note assumed a failed call meant bad
# arguments. A model that had done the discovery correctly then called
# `mcp.mailspring.list_unread_emails` — which does not exist; the real one is
# `list_unread_threads` — and was told the tool "exists and you may call it
# again". It reasonably concluded it could not do the job and said so.


_AVAILABLE = [
    "mcp.mailspring.list_accounts",
    "mcp.mailspring.list_unread_threads",
    "mcp.mailspring.list_threads",
    "shell",
    "recall",
]


def test_an_invented_tool_name_is_not_said_to_exist() -> None:
    from agent.loop import tool_failure_note

    note = tool_failure_note(
        "mcp.mailspring.list_unread_emails", {"accountId": "73b9dacc"}, _AVAILABLE
    )
    assert "exists and you may call it again" not in note
    assert "no tool called" in note


def test_an_invented_name_gets_the_real_one() -> None:
    """The whole point: `list_unread_emails` is one word from the real tool."""
    from agent.loop import tool_failure_note

    note = tool_failure_note(
        "mcp.mailspring.list_unread_emails", {"accountId": "73b9dacc"}, _AVAILABLE
    )
    assert "mcp.mailspring.list_unread_threads" in note


def test_suggestions_come_from_the_same_server_first() -> None:
    from agent.loop import _closest_tools

    assert all(
        t.startswith("mcp.mailspring.")
        for t in _closest_tools("mcp.mailspring.list_unread_emails", _AVAILABLE)
    )


def test_a_name_with_no_close_match_still_gets_somewhere_to_look() -> None:
    from agent.loop import _closest_tools

    found = _closest_tools("mcp.mailspring.wibble", _AVAILABLE)
    assert found
    assert all("list" in t for t in found)


def test_a_real_tool_is_still_told_that_it_is_real() -> None:
    """The two cases need opposite advice; neither may leak into the other."""
    from agent.loop import tool_failure_note

    note = tool_failure_note(
        "mcp.mailspring.list_unread_threads", {"accountId": "user_account_id"}, _AVAILABLE
    )
    assert "no tool called" not in note
    assert "exists and you may call it again" in note
