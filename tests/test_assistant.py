"""Memory, reminders, routing, and MCP auth."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mcpbridge.client import normalize_auth_header, normalize_headers, server_transport
from mcpbridge.forms import (
    mask_token,
    parse_args_field,
    parse_env_lines,
    parse_header_lines,
    server_endpoint,
)
from scheduler.timeparse import describe, parse_when


# ── request routing ──────────────────────────────────────────────────────
#
# Routing used to be a regex heuristic that skipped the agent loop entirely for
# anything that looked like an outward question. It decided for the model, and
# it was wrong in both directions. The model now decides, so what is tested is
# that it has what it needs to decide with: the guidance and the tools.


@pytest.mark.parametrize("tool", ["research_web", "web_search", "web_fetch"])
def test_the_web_tools_are_registered(tool: str) -> None:
    from seams import build_context

    assert build_context().prompt is not None
    from tools.registry import get_registry

    assert get_registry().get(tool) is not None


def test_the_prompt_tells_the_model_when_to_search() -> None:
    from seams import build_context

    system = build_context().prompt.assemble().system
    assert "training data is stale" in system
    assert "research_web" in system


def test_the_prompt_frames_web_content_as_untrusted() -> None:
    """A page the model just read is where an injected instruction would be."""
    from seams import build_context

    system = build_context().prompt.assemble().system
    assert "untrusted" in system.lower()


def test_personal_requests_have_a_memory_path() -> None:
    from tools.registry import get_registry

    for tool in ("recall", "remember", "remind_me"):
        assert get_registry().get(tool) is not None


# ── time parsing ─────────────────────────────────────────────────────────


def test_relative_offsets() -> None:
    now = datetime(2026, 8, 30, 10, 0).astimezone()
    assert parse_when("in 30 minutes", now=now) == now + timedelta(minutes=30)
    assert parse_when("in 2 hours", now=now) == now + timedelta(hours=2)
    assert parse_when("in 3 days", now=now) == now + timedelta(days=3)


def test_tomorrow_defaults_to_nine_am() -> None:
    now = datetime(2026, 8, 30, 22, 0).astimezone()
    got = parse_when("tomorrow", now=now)
    assert (got.day, got.hour, got.minute) == (31, 9, 0)


def test_tomorrow_with_explicit_clock() -> None:
    now = datetime(2026, 8, 30, 10, 0).astimezone()
    got = parse_when("tomorrow at 7:30pm", now=now)
    assert (got.day, got.hour, got.minute) == (31, 19, 30)


def test_bare_low_hour_reads_as_evening() -> None:
    """'remind me at 6' means tonight, not 6am."""
    now = datetime(2026, 8, 30, 10, 0).astimezone()
    assert parse_when("at 6", now=now).hour == 18


def test_bare_time_already_past_rolls_to_tomorrow() -> None:
    now = datetime(2026, 8, 30, 20, 0).astimezone()
    got = parse_when("at 9am", now=now)
    assert (got.day, got.hour) == (31, 9)


def test_named_weekday_moves_forward() -> None:
    now = datetime(2026, 8, 30, 10, 0).astimezone()  # a Sunday
    got = parse_when("friday at 17:00", now=now)
    assert got.weekday() == 4
    assert got > now


def test_iso_timestamps_pass_through() -> None:
    assert parse_when("2026-09-01T08:15:00+00:00").hour in range(24)


def test_unparseable_time_returns_none() -> None:
    assert parse_when("purple monkey dishwasher") is None
    assert parse_when("") is None


def test_describe_is_human_readable() -> None:
    now = datetime(2026, 8, 30, 10, 0).astimezone()
    assert describe(now + timedelta(hours=2), now=now).startswith("today at")
    assert describe(now + timedelta(days=1), now=now).startswith("tomorrow at")


# ── structured memory ────────────────────────────────────────────────────


@pytest.fixture()
def memory_db(tmp_path, monkeypatch):
    """Point the store at a throwaway database."""
    import memory.store as store

    monkeypatch.setattr(store, "_store", None)
    monkeypatch.setattr(store.SessionStore, "__init__", store.SessionStore.__init__)
    fresh = store.SessionStore(tmp_path / "test.db")
    monkeypatch.setattr(store, "_store", fresh)
    return store


def test_save_and_recall_a_fact(memory_db) -> None:
    saved = memory_db.save_memory("Omar's sister's birthday is 3 March", kind="contact")
    assert saved["id"]
    hits = memory_db.search_memories("birthday")
    assert any("3 March" in h["text"] for h in hits)


def test_saving_the_same_fact_twice_updates_rather_than_duplicates(memory_db) -> None:
    first = memory_db.save_memory("Prefers concise answers", kind="preference")
    second = memory_db.save_memory("prefers concise answers!", kind="preference")
    assert first["id"] == second["id"]
    assert len(memory_db.list_memories()) == 1


def test_forget_removes_a_fact(memory_db) -> None:
    saved = memory_db.save_memory("Temporary note")
    assert memory_db.forget_memory(saved["id"]) is True
    assert memory_db.forget_memory(saved["id"]) is False
    assert memory_db.search_memories("Temporary") == []


def test_empty_memory_is_rejected(memory_db) -> None:
    with pytest.raises(ValueError):
        memory_db.save_memory("   ")


def test_digest_is_bounded(memory_db) -> None:
    for i in range(60):
        memory_db.save_memory(f"Fact number {i} about something in the user's life")
    digest = memory_db.memory_digest(max_chars=400)
    assert len(digest) <= 460
    assert digest.startswith("- [")


def test_digest_when_empty(memory_db) -> None:
    assert memory_db.memory_digest() == "(nothing remembered yet)"


def test_inbox_roundtrip(memory_db) -> None:
    memory_db.push_inbox("Keylane", "Background research finished", source="task:1")
    assert len(memory_db.list_inbox()) == 1
    assert memory_db.mark_inbox_read() == 1
    assert memory_db.list_inbox() == []


# ── MCP transport ────────────────────────────────────────────────────────


def test_bare_token_gains_a_bearer_scheme() -> None:
    """Mailspring shows a raw UUID; sending it unprefixed returns 401."""
    assert normalize_auth_header("abc-123") == "Bearer abc-123"


def test_existing_scheme_is_left_alone() -> None:
    assert normalize_auth_header("Bearer abc-123") == "Bearer abc-123"
    assert normalize_auth_header("Basic xyz") == "Basic xyz"


def test_a_pasted_full_header_line_is_unwrapped() -> None:
    assert normalize_auth_header("Authorization: Bearer abc") == "Bearer abc"


def test_blank_tokens_are_dropped() -> None:
    assert normalize_auth_header(None) == ""
    assert normalize_headers({"Authorization": ""}) == {}


def test_only_the_auth_header_is_rewritten() -> None:
    got = normalize_headers({"Authorization": "tok", "X-Trace": "1"})
    assert got == {"Authorization": "Bearer tok", "X-Trace": "1"}


def test_transport_is_inferred_from_the_fields_given() -> None:
    assert server_transport({"url": "http://127.0.0.1:2587/mcp"}) == "http"
    assert server_transport({"command": "npx"}) == "stdio"
    assert server_transport({"transport": "streamable-http", "url": "x"}) == "http"
    assert server_transport({"transport": "stdio", "command": "x"}) == "stdio"


# ── the settings form ────────────────────────────────────────────────────


def test_arguments_split_like_a_shell_line() -> None:
    got = parse_args_field("-y @modelcontextprotocol/server-filesystem /home/user")
    assert got == ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]


def test_a_quoted_argument_with_spaces_stays_one_argument() -> None:
    assert parse_args_field('--root "/home/me/My Files"') == ["--root", "/home/me/My Files"]


def test_commas_still_split_the_way_the_field_used_to() -> None:
    assert parse_args_field("-y, @mcp/fs, /home/user") == ["-y", "@mcp/fs", "/home/user"]


def test_an_unbalanced_quote_falls_back_to_plain_words() -> None:
    assert parse_args_field('--root "/home/me') == ["--root", '"/home/me']


def test_blank_arguments_are_no_arguments() -> None:
    assert parse_args_field("   ") == []
    assert parse_args_field(None) == []


def test_env_lines_need_an_equals_sign() -> None:
    assert parse_env_lines(["TOKEN=abc", "no-equals", "  PATH = /bin "]) == {
        "TOKEN": "abc",
        "PATH": "/bin",
    }


def test_headers_take_either_separator() -> None:
    assert parse_header_lines(["X-Trace: 1", "Authorization=tok", "junk"]) == {
        "X-Trace": "1",
        "Authorization": "tok",
    }


def test_a_token_is_shown_masked_never_whole() -> None:
    """The row proves which token is saved without putting it on screen."""
    srv = {"transport": "http", "url": "http://127.0.0.1:2587/mcp", "auth_header": "abcdef-1234"}
    line = server_endpoint(srv)
    assert "abcdef-1234" not in line
    assert line.startswith("http://127.0.0.1:2587/mcp")
    assert line.endswith("1234")


def test_a_short_token_leaks_no_tail() -> None:
    assert mask_token("short") == "••••"


def test_a_stdio_row_reads_as_the_command_it_runs() -> None:
    srv = {"transport": "stdio", "command": "npx", "args": ["-y", "/home/a b"]}
    assert server_endpoint(srv) == "npx -y '/home/a b'"


# ── reminders survive a restart ──────────────────────────────────────────


@pytest.fixture()
def scheduler_db(tmp_path, monkeypatch):
    """Isolated store plus a stubbed delivery channel."""
    import memory.store as store
    import scheduler.jobs as jobs

    monkeypatch.setattr(store, "_store", store.SessionStore(tmp_path / "sched.db"))
    delivered: list[tuple[str, str]] = []
    monkeypatch.setattr(jobs, "_deliver", lambda title, body, **kw: delivered.append((title, body)))
    return jobs, store, delivered


def test_a_reminder_is_persisted_when_created(scheduler_db) -> None:
    jobs, store, _ = scheduler_db
    result = jobs.create_reminder("call the dentist", "tomorrow at 9am")
    assert result["id"].startswith("remind-")
    rows = store.list_scheduled_tasks()
    assert [r["prompt"] for r in rows] == ["call the dentist"]
    assert rows[0]["run_at"]


def test_a_future_reminder_is_rearmed_after_a_restart(scheduler_db) -> None:
    """The whole point: a reminder set today must still fire after a reboot."""
    jobs, _store, _ = scheduler_db
    created = jobs.create_reminder("stand up", "in 2 hours")

    jobs.get_scheduler().remove_all_jobs()  # simulate the process dying
    assert jobs.get_scheduler().get_jobs() == []

    assert jobs.restore_scheduled_tasks() == 1
    armed = {job.id for job in jobs.get_scheduler().get_jobs()}
    assert created["id"] in armed
    jobs.get_scheduler().remove_all_jobs()


def test_a_recently_missed_reminder_fires_late_exactly_once(scheduler_db) -> None:
    jobs, store, delivered = scheduler_db
    past = (datetime.now().astimezone() - timedelta(hours=2)).isoformat()
    store.upsert_scheduled_task("missed", "reminder", "take the bins out", run_at=past)

    jobs.restore_scheduled_tasks()
    assert any("take the bins out" in body for _title, body in delivered)
    assert any("missed while offline" in body for _title, body in delivered)

    delivered.clear()
    jobs.restore_scheduled_tasks()
    assert delivered == []  # disabled after firing, so it never repeats


def test_a_long_stale_reminder_is_dropped_rather_than_fired(scheduler_db) -> None:
    jobs, store, delivered = scheduler_db
    ancient = (datetime.now().astimezone() - timedelta(days=3)).isoformat()
    store.upsert_scheduled_task("stale", "reminder", "ancient thing", run_at=ancient)

    jobs.restore_scheduled_tasks()
    assert delivered == []
    assert "stale" not in {r["id"] for r in store.list_scheduled_tasks()}


def test_reminders_reject_a_time_in_the_past(scheduler_db) -> None:
    jobs, _store, _ = scheduler_db
    result = jobs.create_reminder("too late", "2020-01-01T00:00:00+00:00")
    assert "error" in result


def test_an_unparseable_time_returns_a_usable_hint(scheduler_db) -> None:
    jobs, _store, _ = scheduler_db
    result = jobs.create_reminder("something", "purple monkey")
    assert "error" in result and "hint" in result


def test_cancelling_a_reminder_removes_it(scheduler_db) -> None:
    jobs, store, _ = scheduler_db
    created = jobs.create_reminder("cancel me", "in 3 hours")
    assert jobs.cancel_task(created["id"]) == {"cancelled": created["id"]}
    assert store.list_scheduled_tasks() == []
    assert "error" in jobs.cancel_task("no-such-task")


def test_a_watcher_needs_a_valid_cron(scheduler_db) -> None:
    jobs, _store, _ = scheduler_db
    assert "error" in jobs.create_watcher("bad", "do a thing", "not a cron")
    ok = jobs.create_watcher("morning-briefing", "summarise today", "0 8 * * 1-5")
    assert ok["id"] == "watch-morning-briefing"
    jobs.get_scheduler().remove_all_jobs()


# ── window placement ─────────────────────────────────────────────────────


def test_auth_header_normalization_is_reused_by_headers() -> None:
    assert normalize_headers({"authorization": "tok"}) == {"authorization": "Bearer tok"}


def test_backend_override_is_read_from_the_environment(monkeypatch) -> None:
    from ui import placement

    monkeypatch.setenv("KEYLANE_BACKEND", "X11")
    assert placement.forced_backend() == "x11"
    monkeypatch.delenv("KEYLANE_BACKEND")
    assert placement.forced_backend() == "auto"


def test_wayland_is_detected_from_either_variable(monkeypatch) -> None:
    from ui import placement

    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert placement.wayland_session() is False
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert placement.wayland_session() is True


def test_floating_geometry_places_each_mode() -> None:
    """The launcher is centred; the orb and HUD sit in the top-right corner."""
    from ui.placement import floating_geometry

    screen = (1920, 1200)
    x, y = floating_geometry("spotlight", 680, 140, screen, 20)
    assert x == (1920 - 680) // 2
    assert 0 < y < 600

    assert floating_geometry("corner", 380, 240, screen, 20) == (1920 - 380 - 20, 20)
    assert floating_geometry("thinking", 56, 56, screen, 20) == (1920 - 56 - 20, 20)


def test_geometry_never_goes_off_the_screen_edge() -> None:
    """A panel wider than the screen pins to 0 rather than a negative offset."""
    from ui.placement import floating_geometry

    for mode in ("spotlight", "corner"):
        x, y = floating_geometry(mode, 680, 140, (320, 240), 20)
        assert x >= 0 and y >= 0
