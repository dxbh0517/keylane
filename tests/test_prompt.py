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
