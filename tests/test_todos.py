"""The todo list: one whole-list write, no ids, no read-then-mutate."""

from __future__ import annotations

import json

import pytest

from tools.todos import load_todos, render_todos, save_todos, todo_write


@pytest.fixture(autouse=True)
def todo_file(tmp_path, monkeypatch):
    path = tmp_path / "todos.json"
    monkeypatch.setattr("tools.todos.TODOS_PATH", path)
    return path


def test_a_write_replaces_the_whole_list() -> None:
    todo_write([{"content": "one", "status": "pending"}])
    todo_write([{"content": "two", "status": "in_progress"}])
    assert load_todos() == [{"content": "two", "status": "in_progress"}]


def test_the_result_reports_what_is_left() -> None:
    payload = json.loads(
        todo_write(
            [
                {"content": "a", "status": "completed"},
                {"content": "b", "status": "in_progress"},
                {"content": "c", "status": "pending"},
            ]
        )
    )
    assert payload["remaining"] == 2


def test_a_bare_string_is_accepted_as_a_pending_todo() -> None:
    todo_write(["buy milk"])
    assert load_todos() == [{"content": "buy milk", "status": "pending"}]


@pytest.mark.parametrize(
    "todos, message",
    [
        ("not a list", "array"),
        ([{"content": "", "status": "pending"}], "non-empty"),
        ([{"content": "x", "status": "doing"}], "status must be"),
        ([42], "object with content"),
    ],
)
def test_a_malformed_list_is_refused_with_a_reason(todos, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        todo_write(todos)


def test_an_empty_list_clears_everything() -> None:
    todo_write([{"content": "one", "status": "pending"}])
    todo_write([])
    assert load_todos() == []


def test_the_old_title_and_done_shape_still_loads(todo_file) -> None:
    """An existing todos.json must not be lost to the new shape."""
    todo_file.write_text(
        json.dumps(
            [
                {"id": "todo-1", "title": "buy milk", "done": False},
                {"id": "todo-2", "title": "call Sam", "done": True},
            ]
        ),
        encoding="utf-8",
    )
    assert load_todos() == [
        {"content": "buy milk", "status": "pending"},
        {"content": "call Sam", "status": "completed"},
    ]


def test_a_corrupt_file_reads_as_empty(todo_file) -> None:
    todo_file.write_text("{ not json", encoding="utf-8")
    assert load_todos() == []


def test_the_list_is_rendered_for_the_context_block() -> None:
    todo_write(
        [
            {"content": "read the docs", "status": "completed"},
            {"content": "write the code", "status": "in_progress"},
            {"content": "run the tests", "status": "pending"},
        ]
    )
    rendered = render_todos()
    assert "[x] read the docs" in rendered
    assert "[~] write the code" in rendered
    assert "[ ] run the tests" in rendered


def test_an_empty_list_contributes_no_context() -> None:
    save_todos([])
    assert render_todos() == ""


def test_the_model_never_has_to_read_before_writing() -> None:
    """The list is in the context block, so there is no list tool to forget."""
    from seams import build_context
    from tools.registry import get_registry

    build_context()
    assert get_registry().get("todo_write") is not None
    assert get_registry().get("todos_list") is None
    assert get_registry().get("todos_add") is None
