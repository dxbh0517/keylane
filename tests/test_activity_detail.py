"""What the control panel needs to show a task: its steps, and what it is asking.

A one-line summary is enough to know something is happening. It is not enough
to approve an action, or to follow one — both need the arguments and the trail.
"""

from __future__ import annotations

import pytest

from app.activity import MAX_STEPS, ActivityBus


@pytest.fixture
def bus():
    return ActivityBus()


async def _start(bus: ActivityBus, task_id: str = "t1") -> None:
    await bus.start_task(task_id, "do the thing", worker="assistant")


# ------------------------------------------------------------------- steps


@pytest.mark.asyncio
async def test_steps_are_recorded_in_order_with_their_detail(bus):
    await _start(bus)

    await bus.record_step(
        "t1",
        thought="heuristic match",
        tool="open_application",
        arguments={"application": "firefox"},
        observation="Launched Firefox.",
    )
    await bus.record_step("t1", tool="read_file", arguments={"path": "/etc/hostname"})

    task = bus.snapshot().active[0]
    assert [s.index for s in task.steps] == [1, 2]
    assert task.steps[0].tool == "open_application"
    assert task.steps[0].arguments == {"application": "firefox"}
    assert task.steps[0].observation == "Launched Firefox."
    assert task.steps[0].thought == "heuristic match"


@pytest.mark.asyncio
async def test_a_failed_step_is_marked_not_ok(bus):
    await _start(bus)

    await bus.record_step("t1", tool="read_file", observation="No such file", ok=False)

    assert bus.snapshot().active[0].steps[0].ok is False


@pytest.mark.asyncio
async def test_the_latest_step_becomes_the_row_summary(bus):
    await _start(bus)

    await bus.record_step("t1", tool="open_application", ok=True)

    assert bus.snapshot().active[0].step == "open_application: ok"


@pytest.mark.asyncio
async def test_a_runaway_loop_cannot_grow_the_snapshot_without_limit(bus):
    # Every SSE frame carries the steps, so an unbounded list would be a leak
    # pushed to every connected client.
    await _start(bus)

    for _ in range(MAX_STEPS + 10):
        await bus.record_step("t1", tool="loop")

    assert len(bus.snapshot().active[0].steps) == MAX_STEPS


@pytest.mark.asyncio
async def test_recording_a_step_on_an_unknown_task_is_a_no_op(bus):
    await bus.record_step("nope", tool="open_application")

    assert bus.snapshot().active == []


# -------------------------------------------------------------- confirmation


@pytest.mark.asyncio
async def test_a_waiting_task_carries_what_it_wants_to_run(bus):
    await _start(bus)

    await bus.update_task(
        "t1",
        status="waiting_confirmation",
        pending_tool="run_command",
        pending_arguments={"command": "rm -rf /tmp/demo"},
    )

    task = bus.snapshot().active[0]
    assert task.pending_tool == "run_command"
    # Approving a bare tool name is not consent — the arguments must survive.
    assert task.pending_arguments == {"command": "rm -rf /tmp/demo"}


@pytest.mark.asyncio
async def test_a_waiting_task_counts_as_needing_attention(bus):
    await _start(bus)

    await bus.update_task("t1", status="waiting_confirmation", pending_tool="x")

    assert bus.snapshot().needs_attention == 1


@pytest.mark.asyncio
async def test_answering_the_question_clears_it(bus):
    await _start(bus)
    await bus.update_task(
        "t1", status="waiting_confirmation", pending_tool="run_command",
        pending_arguments={"command": "ls"},
    )

    await bus.update_task("t1", status="running", clear_pending=True)

    task = bus.snapshot().active[0]
    assert task.pending_tool is None
    assert task.pending_arguments == {}


@pytest.mark.asyncio
async def test_steps_survive_into_history_so_a_finished_task_can_be_reviewed(bus):
    await _start(bus)
    await bus.record_step("t1", tool="open_application", observation="Launched Firefox.")

    await bus.update_task("t1", status="completed")

    recent = bus.snapshot().recent
    assert recent[0].task_id == "t1"
    assert recent[0].steps[0].observation == "Launched Firefox."
