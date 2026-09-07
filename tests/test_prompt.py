"""Prompt assembly: a stable prefix, a dynamic block, and no drift."""

from __future__ import annotations

import pytest

from seams.prompt import (
    CONTEXT_OPEN,
    PromptError,
    SystemPrompt,
    latest_context_digest,
)


@pytest.fixture()
def prompt():
    return SystemPrompt()


def test_sections_are_ordered_by_their_named_placement(prompt) -> None:
    prompt.section("output", "LAST")
    prompt.section("identity", "FIRST")
    prompt.section("web", "MIDDLE")
    assert prompt.assemble().system == "FIRST\n\nMIDDLE\n\nLAST"


def test_an_unregistered_tool_contributes_no_guidance(prompt) -> None:
    """This is the whole point: the prompt cannot promise a missing tool."""
    prompt.section("identity", "You are Keylane.")
    dispose = prompt.section("web", "Use web_search for current information.")
    assert "web_search" in prompt.assemble().system
    dispose()
    assert "web_search" not in prompt.assemble().system


def test_a_section_may_be_computed_at_assembly(prompt) -> None:
    tools = ["recall"]
    prompt.section("tools", lambda: f"## Tools\n{', '.join(tools)}")
    assert "recall" in prompt.assemble().system
    tools.append("web_search")
    assert "web_search" in prompt.assemble().system


def test_an_empty_section_contributes_nothing(prompt) -> None:
    prompt.section("identity", "You are Keylane.")
    prompt.section("web", "   ")
    assert prompt.assemble().system == "You are Keylane."


def test_variables_are_interpolated(prompt) -> None:
    prompt.variable("assistant_name", lambda: "Keylane")
    prompt.section("identity", "You are {{assistant_name}}.")
    assert prompt.assemble().system == "You are Keylane."


def test_an_undefined_variable_fails_loudly(prompt) -> None:
    """A hole that renders as an empty string is a hole nobody notices."""
    prompt.section("identity", "You are {{nobody}}.")
    with pytest.raises(PromptError, match="nobody"):
        prompt.assemble()


def test_an_invalid_variable_name_is_rejected(prompt) -> None:
    with pytest.raises(PromptError):
        prompt.variable("Bad-Name", lambda: "x")


def test_an_unknown_placement_name_is_rejected(prompt) -> None:
    with pytest.raises(PromptError, match="placement"):
        prompt.section("invented", "text")


def test_an_explicit_order_bypasses_the_table(prompt) -> None:
    prompt.section("invented", "text", order=42)
    assert prompt.assemble().system == "text"


# ── the static / dynamic split ───────────────────────────────────────────


def test_volatile_facts_stay_out_of_the_system_message(prompt) -> None:
    """The system message must be byte-identical across turns to be cacheable."""
    clock = ["00:01"]
    prompt.section("identity", "You are Keylane.")
    prompt.context("now", lambda: f"Local time: {clock[0]}")

    first = prompt.assemble()
    clock[0] = "00:02"
    second = prompt.assemble()

    assert first.system == second.system
    assert first.context != second.context


def test_the_dynamic_block_is_wrapped_so_it_can_be_found_again(prompt) -> None:
    prompt.section("identity", "You are Keylane.")
    prompt.context("now", lambda: "Local time: noon")
    assembly = prompt.assemble()
    assert assembly.context.startswith(CONTEXT_OPEN)
    assert "Local time: noon" in assembly.context


def test_no_contexts_means_no_block(prompt) -> None:
    prompt.section("identity", "You are Keylane.")
    assert prompt.assemble().context == ""


def test_the_digest_finds_the_newest_block_still_in_history(prompt) -> None:
    prompt.section("identity", "x")
    prompt.context("now", lambda: "Local time: noon")
    assembly = prompt.assemble()

    history = [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": assembly.context},
        {"role": "assistant", "content": "hi"},
    ]
    assert latest_context_digest(history) == assembly.context_digest


def test_an_unchanged_context_is_not_re_emitted(prompt) -> None:
    prompt.section("identity", "x")
    prompt.context("now", lambda: "Local time: noon")
    assembly = prompt.assemble()
    history = [{"role": "user", "content": assembly.context}]
    # Same content, so the loop's guard finds a match and appends nothing.
    assert latest_context_digest(history) == prompt.assemble().context_digest


def test_a_changed_context_is_re_emitted(prompt) -> None:
    clock = ["noon"]
    prompt.section("identity", "x")
    prompt.context("now", lambda: f"Local time: {clock[0]}")
    history = [{"role": "user", "content": prompt.assemble().context}]
    clock[0] = "midnight"
    assert latest_context_digest(history) != prompt.assemble().context_digest


def test_history_without_a_block_has_no_digest() -> None:
    assert latest_context_digest([{"role": "user", "content": "hello"}]) == ""


# ── the composed prompt ──────────────────────────────────────────────────


def test_the_composed_prompt_describes_only_registered_tools() -> None:
    from seams import build_context

    ctx = build_context()
    assembly = ctx.prompt.assemble()
    for name in ("recall", "remember", "research_web", "remind_me"):
        assert name in assembly.system
    assert "{{" not in assembly.system


def test_the_composed_system_prompt_is_stable_across_assemblies() -> None:
    from seams import build_context

    ctx = build_context()
    assert ctx.prompt.assemble().system == ctx.prompt.assemble().system


# ── fitting a budget ─────────────────────────────────────────────────────


def test_a_prompt_within_budget_keeps_everything(prompt) -> None:
    prompt.section("identity", "You are Keylane.")
    prompt.section("web", "Search guidance.", required=False)
    assert "Search guidance." in prompt.assemble(budget_chars=10_000).system


def test_optional_guidance_is_dropped_before_the_output_contract(prompt) -> None:
    """Truncating the string would cut exactly the parts that must survive."""
    prompt.section("identity", "IDENTITY")
    prompt.section("web", "W" * 400, required=False)
    prompt.section("memory", "M" * 400, required=False)
    prompt.section("output", "OUTPUT CONTRACT")
    prompt.section("tool_format", "TOOL FORMAT")

    system = prompt.assemble(budget_chars=120).system
    assert "IDENTITY" in system
    assert "OUTPUT CONTRACT" in system
    assert "TOOL FORMAT" in system
    assert "W" * 400 not in system
    assert "M" * 400 not in system


def test_the_most_specialised_guidance_goes_first(prompt) -> None:
    prompt.section("identity", "IDENTITY")
    prompt.section("memory", "M" * 200, required=False)
    prompt.section("subagent", "S" * 200, required=False)

    # Enough room for identity plus one guidance paragraph.
    system = prompt.assemble(budget_chars=260).system
    assert "M" * 200 in system
    assert "S" * 200 not in system


def test_required_sections_survive_even_when_they_do_not_fit(prompt) -> None:
    """A prompt with no tool-call format is worse than one over budget."""
    prompt.section("identity", "I" * 500)
    prompt.section("tool_format", "T" * 500)
    system = prompt.assemble(budget_chars=100).system
    assert "I" * 500 in system
    assert "T" * 500 in system


def test_a_zero_budget_means_unbounded(prompt) -> None:
    prompt.section("identity", "IDENTITY")
    prompt.section("web", "W" * 5000, required=False)
    assert "W" * 5000 in prompt.assemble(budget_chars=0).system


def test_the_composed_prompt_fits_an_npu_budget() -> None:
    """The real prompt against the real limit — this is what threw in production."""
    from npu.limits import CHARS_PER_TOKEN, NPU_MAX_PROMPT_TOKENS, npu_prompt_budget_chars
    from seams import build_context

    budget = npu_prompt_budget_chars()
    system = build_context().prompt.assemble(budget_chars=budget).system
    assert len(system) <= budget
    assert len(system) / CHARS_PER_TOKEN < NPU_MAX_PROMPT_TOKENS
    # The non-negotiable parts are still there.
    assert "<tool_call>" in system
    assert "First line is the answer" in system


# ── the tool index ───────────────────────────────────────────────────────


def test_the_tool_index_bounds_each_description() -> None:
    """Thirty full descriptions is most of an NPU pipeline's whole budget."""
    from tools.registry import Tool, ToolRegistry

    reg = ToolRegistry()
    reg.register(
        Tool(
            name="verbose",
            description="Do the thing. " + "And then a great deal more detail. " * 20,
            parameters={"type": "object", "properties": {"a": {"type": "string"}}},
            handler=lambda a: a,
        )
    )
    line = reg.describe_for_prompt()
    assert line.startswith("- verbose(a): Do the thing.")
    assert len(line) < 200


def test_a_short_description_is_left_alone() -> None:
    from tools.registry import Tool, ToolRegistry

    reg = ToolRegistry()
    reg.register(
        Tool(
            name="recall",
            description="Search saved memories.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "",
        )
    )
    assert reg.describe_for_prompt() == "- recall(): Search saved memories."


def test_a_long_first_sentence_is_truncated_not_kept_whole() -> None:
    from tools.registry import ToolRegistry

    summary = ToolRegistry._summarize("x" * 400)
    assert len(summary) <= ToolRegistry.DESCRIPTION_CAP + 1
    assert summary.endswith("…")


def test_the_required_prompt_fits_a_modest_npu_pipeline() -> None:
    """A user trading compile time for context must still get a usable prompt."""
    from seams import build_context

    required = build_context().prompt.assemble(budget_chars=1).system
    assert len(required) / 2.8 < 2048
    assert "<tool_call>" in required


# ── the preamble can crowd out the conversation without failing ──────────
#
# Connecting Mailspring's MCP server registered 21 tools, each a line in the
# tool index — which is a *required* section, because a model that loses it
# cannot call anything. At the old 4096-token NPU budget the required floor was
# 8171 of 9318 characters: every optional section dropped, and 342 characters
# left for the conversation. Nothing failed. The model simply had the tools and
# no room to be asked to use them.


def _prompt_with(required_chars: int):
    from seams.prompt import SystemPrompt

    prompt = SystemPrompt()
    prompt.section("identity", "i" * required_chars, required=True)
    prompt.section("memory", "m" * 400, required=False)
    prompt.section("web", "w" * 400, required=False)
    return prompt


def test_a_roomy_prompt_is_not_starved() -> None:
    assembly = _prompt_with(200).assemble(budget_chars=8000)
    assert not assembly.starved
    assert assembly.headroom_chars > 0


def test_a_preamble_that_fills_the_budget_is_starved() -> None:
    """The signal that was missing: nothing raises, so nothing said so."""
    assembly = _prompt_with(7900).assemble(budget_chars=8000)
    assert assembly.starved
    assert assembly.headroom_share < 0.25


def test_optional_sections_are_dropped_before_the_required_ones() -> None:
    """Losing the tool index would leave the model unable to call anything."""
    assembly = _prompt_with(7000).assemble(budget_chars=7400)
    assert "i" * 100 in assembly.system
    assert "m" * 100 not in assembly.system


def test_no_budget_means_no_verdict() -> None:
    """Nothing imposed a limit, so nothing is being starved by one."""
    assembly = _prompt_with(9000).assemble()
    assert not assembly.starved
    assert assembly.headroom_share == 1.0


# The required floor measured with 55 tools registered — Keylane's own plus
# Mailspring's 21. It grows with every tool, and it is what the budget has to
# clear.
MEASURED_FLOOR_CHARS = 8171


def test_the_npu_budget_leaves_room_for_a_real_tool_set() -> None:
    """4096 tokens stopped being enough the moment an MCP server was added.

    Stated against the share the conversation must keep rather than a round
    multiple, because the two constants pull against each other: room for a
    longer reply is taken out of the room for the prompt, and raising one
    without checking the other is how this budget gets crossed a third time.
    """
    from npu.limits import (
        MIN_CONVERSATION_SHARE,
        NPU_MAX_PROMPT_TOKENS,
        npu_prompt_budget_chars,
    )

    assert NPU_MAX_PROMPT_TOKENS >= 8192
    budget = npu_prompt_budget_chars()
    headroom = budget - MEASURED_FLOOR_CHARS
    assert headroom > 0, "the tool index alone does not fit"
    assert headroom / budget >= MIN_CONVERSATION_SHARE


def test_a_reasoning_model_gets_room_to_finish() -> None:
    """Measured: ~400 tokens of reasoning before the tool call was emitted.

    512 left about a hundred tokens of slack, which one turn of history
    consumes — and the failure is silence rather than a short answer, because
    reasoning is stripped before the user sees it.
    """
    from npu.limits import MAX_REPLY_TOKENS

    assert MAX_REPLY_TOKENS >= 1024


def test_the_reply_is_declared_to_the_pipeline_not_taken_from_the_prompt() -> None:
    """The prompt window and the response allowance are separate on the NPU.

    Taking the reply out of the prompt window reserves the same tokens twice.
    It cost 4000 characters of prompt, which with an MCP server connected was
    the difference between 33% of the budget left for the conversation and 17%.
    """
    from npu.limits import MAX_REPLY_TOKENS, RESERVE_TOKENS
    from npu.pipeline_config import pipeline_init_kwargs

    assert RESERVE_TOKENS < MAX_REPLY_TOKENS, "the reserve is scaffolding, not the reply"
    npu = pipeline_init_kwargs("NPU", None, "llm")
    assert npu["MIN_RESPONSE_LEN"] == MAX_REPLY_TOKENS
    assert npu["MAX_PROMPT_LEN"] >= 8192


def test_a_vlm_on_the_npu_declares_the_same_window() -> None:
    from npu.limits import MAX_REPLY_TOKENS
    from npu.pipeline_config import pipeline_init_kwargs

    props = pipeline_init_kwargs("NPU", None, "vlm")["config"]["DEVICE_PROPERTIES"]["NPU"]
    assert props["MIN_RESPONSE_LEN"] == MAX_REPLY_TOKENS


def test_only_the_npu_compiles_a_window_in() -> None:
    """CPU and GPU are bounded by patience, not by a compiled graph."""
    from npu.pipeline_config import pipeline_init_kwargs

    assert pipeline_init_kwargs("CPU", None, "llm") == {}


def test_the_budget_clears_the_floor_with_an_mcp_server_connected() -> None:
    """55 tools is a real configuration: Keylane's own plus Mailspring's 21."""
    from npu.limits import MIN_CONVERSATION_SHARE, npu_prompt_budget_chars

    budget = npu_prompt_budget_chars()
    assert (budget - MEASURED_FLOOR_CHARS) / budget >= MIN_CONVERSATION_SHARE
