"""Tool calls in the dialects small models actually emit.

Keylane's own prompt asks for `<tool_call>{json}</tool_call>`, and a model that
was trained on that writes it. MiniCPM5 was not: its chat template teaches
`<function name="x"><param name="y">v</param></function>`, and it writes that
whatever the prompt says. A parser that knows only the first dialect does not
fail loudly — the call falls through as prose and the model looks as though it
chose not to use its tools.
"""

from __future__ import annotations

import pytest

from agent.tools_parse import has_tool_call_markup, parse_tool_call, strip_tool_call
from npu.thinking import sanitize_response

# ── the MiniCPM5 dialect ─────────────────────────────────────────────────


def test_an_xml_function_call_is_parsed() -> None:
    call = parse_tool_call(
        '<function name="web_search"><param name="question">rain today</param></function>'
    )
    assert call == {"name": "web_search", "arguments": {"question": "rain today"}}


def test_a_call_with_no_parameters_is_still_a_call() -> None:
    assert parse_tool_call('<function name="inbox_list"></function>') == {
        "name": "inbox_list",
        "arguments": {},
    }


def test_a_self_closing_call_is_a_call() -> None:
    assert parse_tool_call('<function name="screen_clear"/>') == {
        "name": "screen_clear",
        "arguments": {},
    }


def test_several_parameters_all_arrive() -> None:
    call = parse_tool_call(
        '<function name="screen_annotate">'
        '<param name="caption">click here</param>'
        '<param name="seconds">4</param>'
        "</function>"
    )
    assert call["arguments"] == {"caption": "click here", "seconds": 4}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("4", 4),
        ("4.5", 4.5),
        ("true", True),
        ("false", False),
        ("null", None),
        ('["a", "b"]', ["a", "b"]),
        ("plain words", "plain words"),
        ("2026-09-06", "2026-09-06"),
    ],
)
def test_xml_values_are_coerced_to_the_type_the_schema_wants(raw, expected) -> None:
    """XML carries no types, so "4" would otherwise reach a number field."""
    call = parse_tool_call(f'<function name="t"><param name="v">{raw}</param></function>')
    assert call["arguments"]["v"] == expected


def test_a_cdata_value_is_taken_verbatim() -> None:
    """CDATA is used precisely because the value contains markup or newlines."""
    call = parse_tool_call(
        '<function name="memory_write"><param name="content">'
        "<![CDATA[line one\nline <two> & three]]>"
        "</param></function>"
    )
    assert call["arguments"]["content"] == "line one\nline <two> & three"


def test_a_cdata_number_stays_a_string() -> None:
    call = parse_tool_call(
        '<function name="t"><param name="v"><![CDATA[4]]></param></function>'
    )
    assert call["arguments"]["v"] == "4"


def test_single_quoted_attributes_work() -> None:
    assert parse_tool_call("<function name='ping'></function>")["name"] == "ping"


def test_prose_around_the_call_does_not_break_it() -> None:
    call = parse_tool_call(
        'I will look that up.\n<function name="web_search">'
        '<param name="question">tide times</param></function>\nOne moment.'
    )
    assert call["name"] == "web_search"


# ── the dialect must not be mistaken for prose, or prose for it ──────────


def test_the_dialect_is_recognised_as_tool_markup() -> None:
    assert has_tool_call_markup('<function name="x"></function>')


def test_prose_about_functions_is_not_a_tool_call() -> None:
    """"<function" alone is not enough; the name attribute is what makes it one."""
    assert not has_tool_call_markup("A function is a mapping from inputs to outputs.")
    assert parse_tool_call("Define a function that returns 4.") is None


def test_the_existing_json_dialect_still_works() -> None:
    assert parse_tool_call('<tool_call>{"name": "recall", "arguments": {}}</tool_call>') == {
        "name": "recall",
        "arguments": {},
    }


def test_a_json_dialect_call_wins_over_a_stray_object() -> None:
    call = parse_tool_call(
        '<tool_call>{"name": "recall", "arguments": {"q": "x"}}</tool_call>\n{"name": "other"}'
    )
    assert call["name"] == "recall"


# ── the user must never read the markup ──────────────────────────────────


def test_the_markup_is_stripped_from_what_the_user_sees() -> None:
    text = 'Looking now. <function name="web_search"><param name="question">x</param></function>'
    assert strip_tool_call(text) == "Looking now."


def test_an_unterminated_call_is_stripped_too() -> None:
    """A stream cut mid-markup must not leave a dangling tag in the answer."""
    assert strip_tool_call('Fine. <function name="web_search"><param name="q">x') == "Fine."


def test_the_answer_card_never_shows_the_dialect() -> None:
    """sanitize_response is the HUD's own path and strips markup separately."""
    text = 'Here you go. <function name="notify_user"><param name="title">hi</param></function>'
    assert sanitize_response(text) == "Here you go."


def test_a_thinking_block_and_a_call_are_both_removed() -> None:
    text = (
        "<think>I should search for this.</think>\n"
        '<function name="web_search"><param name="question">x</param></function>'
    )
    assert sanitize_response(text) == ""


# ── the field a model uses for the tool's name ───────────────────────────
#
# Observed mid-turn from Qwen3, having used `name` correctly on the call
# before: {"tool_call": "mcp.mailspring.list_folders", "arguments": {...}}.
# Refusing that spelling does not produce an error — the JSON is handed to the
# user as though it were the answer.


@pytest.mark.parametrize(
    "raw",
    [
        '{"name": "inbox_list", "arguments": {}}',
        '{"tool_call": "inbox_list", "arguments": {}}',
        '{"tool": "inbox_list", "args": {}}',
        '{"function": "inbox_list", "parameters": {}}',
        '{"tool_name": "inbox_list", "input": {}}',
    ],
)
def test_the_name_is_read_whatever_the_model_calls_the_field(raw: str) -> None:
    assert parse_tool_call(raw) == {"name": "inbox_list", "arguments": {}}


def test_arguments_are_read_whatever_the_model_calls_them() -> None:
    call = parse_tool_call('{"tool_call": "web_search", "params": {"question": "tides"}}')
    assert call["arguments"] == {"question": "tides"}


def test_ordinary_json_is_still_not_a_tool_call() -> None:
    """The looser the name matching, the more this one matters."""
    assert parse_tool_call('{"unrelated": "json", "value": 3}') is None
    assert parse_tool_call('{"name": ""}') is None


def test_a_json_object_of_prose_is_not_a_call() -> None:
    assert parse_tool_call('{"answer": "you have three unread emails"}') is None
