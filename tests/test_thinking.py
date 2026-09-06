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
