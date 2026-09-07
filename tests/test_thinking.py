from npu.thinking import OutputStreamFilter, ThinkingStreamFilter, extract_user_answer, strip_thinking, sanitize_response

_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "</" + "think" + ">"


def test_strip_redacted_thinking_block():
    raw = "<think>\nsecret\n</think>\n\nHello world"
    assert strip_thinking(raw) == "Hello world"


def test_strip_orphan_redacted_thinking_close():
    raw = (
        "The user is asking about Civ6.\n</think>\n\n"
        "Catapult upgrades to Bombard."
    )
    assert strip_thinking(raw) == "Catapult upgrades to Bombard."


def test_extract_user_answer_strips_tool_blocks():
    raw = (
        "The user is asking.\n</think>\n\n"
        "<tool_call>{\"name\": \"research_web\"}</tool_call>\n\n"
        "Based on sources [1], Catapult upgrades to Bombard.\n\n"
        "Sources\n[1] Wiki — https://example.com"
    )
    out = extract_user_answer(raw)
    assert "tool_call" not in out.lower()
    assert "Catapult" in out
    assert "Sources" in out


def test_strip_qwen3_think_block():
    raw = f"{_THINK_OPEN}\nsecret reasoning\n{_THINK_CLOSE}\n\nHello world"
    assert strip_thinking(raw) == "Hello world"


def test_strip_thinking_process_prefix():
    raw = "Thinking Process:\n\n1. Reason\n\nFinal answer here"
    assert strip_thinking(raw) == "Final answer here"


def test_sanitize_response_strips_tool_call():
    raw = (
        f"{_THINK_OPEN}\nhidden\n{_THINK_CLOSE}\n"
        "Answer text <tool_call>{\"name\": \"web_search\"}</tool_call>"
    )
    assert sanitize_response(raw) == "Answer text"


def test_stream_filter_hides_thinking_until_end():
    filt = ThinkingStreamFilter()
    assert filt.feed("<think>\nsecret") == ""
    assert filt.feed("\n</think>\n\nHello") == "Hello"


def test_stream_filter_hides_qwen3_think_block():
    filt = ThinkingStreamFilter()
    assert filt.feed(_THINK_OPEN + "\nsecret") == ""
    assert filt.feed("\n" + _THINK_CLOSE + "\n\nHello") == "Hello"


def test_stream_filter_plain_text_passes_through():
    filt = ThinkingStreamFilter()
    assert filt.feed("Hello ") == "Hello "
    assert filt.feed("world") == "world"


def test_output_stream_filter_holds_tool_markup():
    filt = OutputStreamFilter()
    assert filt.feed("Here is ") == "Here is "
    assert filt.feed("<tool_call>{") == ""
    assert filt.feed('"name": "web_search"}') == ""
    assert filt.flush() == ""


# ── a reply spent entirely on reasoning ──────────────────────────────────
#
# A reasoning model spends its budget thinking before it answers, and the
# reasoning is stripped before the user sees it — so a cap it cannot finish
# inside produces an empty reply rather than a short one. The narrow check
# only caught the unterminated case; a model that closes its block and then
# runs out fell through to "I could not produce a response", which is the one
# thing that did not happen.


def test_an_unterminated_reasoning_block_is_caught() -> None:
    from npu.thinking import reasoned_without_answering

    assert reasoned_without_answering("<think>\nstill working on it and then the budget")


def test_a_closed_block_with_no_answer_after_it_is_caught() -> None:
    """The case that was reported as "could not produce a response"."""
    from npu.thinking import reasoned_without_answering

    assert reasoned_without_answering("<think>\nI have decided what to do.\n</think>\n\n")


def test_reasoning_followed_by_an_answer_is_not_caught() -> None:
    from npu.thinking import reasoned_without_answering

    assert not reasoned_without_answering("<think>\nthinking\n</think>\n\nYou have 3 unread.")


def test_a_plain_empty_reply_is_not_blamed_on_reasoning() -> None:
    """No reasoning happened, so saying it was spent thinking would be untrue."""
    from npu.thinking import reasoned_without_answering

    assert not reasoned_without_answering("")
    assert not reasoned_without_answering("   \n ")


def test_a_reply_that_is_only_a_tool_call_is_not_blamed_on_reasoning() -> None:
    """It sanitizes to nothing too, and it is the loop's business, not this."""
    from npu.thinking import reasoned_without_answering

    assert not reasoned_without_answering(
        '<tool_call>\n{"name": "mcp.mailspring.list_accounts", "arguments": {}}\n</tool_call>'
    )
