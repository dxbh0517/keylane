"""Task orchestration: route → confirm → execute → verify → retry."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.activity import get_activity_bus
from app.assistant import AssistantOutcome, get_assistant
from app.assistant_settings import load_assistant_settings
from app.config import AppConfig, get_config
from app.permissions import PermissionError_, is_local_only, validate_route
from app.planner import build_plan, needs_multi_step
from app.plugins.registry import get_plugin_registry
from app.router import RouterService
from app.schemas import (
    ChatRequest,
    RouteDecision,
    TaskRecord,
    TaskResponse,
    TaskStatus,
    WorkerEvidence,
    WorkerResult,
)
from app.verifier import VerifierService

logger = logging.getLogger(__name__)


class TaskStore:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = asyncio.Lock()

    async def put(self, task: TaskRecord) -> None:
        async with self._lock:
            self._tasks[task.task_id] = task

    async def get(self, task_id: str) -> TaskRecord | None:
        async with self._lock:
            return self._tasks.get(task_id)

    async def update(self, task: TaskRecord) -> None:
        task.touch()
        async with self._lock:
            self._tasks[task.task_id] = task


class GatewayOrchestrator:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.registry = get_plugin_registry(self.config)
        self.router = RouterService(self.config, self.registry)
        self.verifier = VerifierService(self.config)
        self.assistant = get_assistant(self.config)
        self.activity = get_activity_bus()
        self.store = TaskStore()
        self._cancel: set[str] = set()

    def _to_response(self, task: TaskRecord) -> TaskResponse:
        return TaskResponse(
            task_id=task.task_id,
            status=task.status,
            worker=task.worker,
            result=task.result,
            route=task.route,
            verification=task.verification,
            requires_confirmation=task.status == TaskStatus.WAITING_CONFIRMATION,
            error=task.error,
            attempt=task.attempt,
            assistant_steps=task.assistant_steps,
            canvas=task.canvas,
            pending_tool=task.pending_tool,
            pending_arguments=task.pending_arguments,
        )

    async def route_only(
        self,
        message: str,
        *,
        project: str | None = None,
        local_only: bool | None = None,
    ) -> RouteDecision:
        return await self.router.route(
            message, project=project, local_only=local_only
        )

    async def approve(self, task_id: str) -> TaskResponse:
        """Let a task waiting on confirmation proceed.

        Separate from :meth:`chat` because saying yes needs no message — the
        panel approves a task it is looking at, and has nothing to re-send.
        """
        task = await self.store.get(task_id)
        if task is None:
            return TaskResponse(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error="Unknown task_id",
            )
        if task.status != TaskStatus.WAITING_CONFIRMATION:
            # Already answered, or never asked. Report where it actually is
            # rather than running it a second time.
            return self._to_response(task)
        if task.pending_tool:
            return await self._resume_assistant(task)
        return await self._execute_with_retries(task)

    async def chat(self, request: ChatRequest) -> TaskResponse:
        # Resume confirmation
        if request.task_id and request.confirmed:
            return await self.approve(request.task_id)

        task = TaskRecord(
            message=request.message,
            project=request.project,
            max_retries=self.config.gateway.max_retries,
            status=TaskStatus.ROUTING,
        )
        await self.store.put(task)
        await self.activity.start_task(task.task_id, request.message)

        local_only = is_local_only(self.config, request.local_only)

        # The NPU assistant gets first refusal: it may finish the job with its
        # own tools, or delegate to a worker and follow up on the result.
        assistant_response = await self._try_assistant(task, local_only=local_only)
        if assistant_response is not None:
            return assistant_response

        try:
            if needs_multi_step(request.message) and request.project:
                plan = build_plan(request.message, request.project)
                task.plan = plan
                # Use first step as the primary route for confirmation UX
                decision = plan.steps[0].decision
                # If any step needs confirmation, gate the whole plan
                if any(s.decision.requires_confirmation for s in plan.steps):
                    # Prefer showing the coding step confirmation
                    for step in plan.steps:
                        if step.decision.requires_confirmation:
                            decision = step.decision
                            break
                # A plan is built locally, so its steps still have to clear the
                # same policy gate a routed decision does.
                for step in plan.steps:
                    validate_route(
                        step.decision, local_only=local_only, config=self.config
                    )
            else:
                decision = await self.router.route(
                    request.message,
                    project=request.project,
                    local_only=request.local_only,
                )
                task.plan = None

            task.route = decision
            task.worker = decision.worker
            task.local_only = local_only
        except PermissionError_ as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            await self.store.update(task)
            await self.activity.update_task(
                task.task_id, status="failed", error=str(exc)
            )
            return self._to_response(task)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Routing failed")
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            await self.store.update(task)
            await self.activity.update_task(
                task.task_id, status="failed", error=str(exc)
            )
            return self._to_response(task)

        await self.activity.update_task(
            task.task_id, status="routing", worker=decision.worker
        )

        if decision.requires_confirmation and not request.confirmed:
            task.status = TaskStatus.WAITING_CONFIRMATION
            task.result = (
                f"{decision.worker} wants to run '{decision.action}' "
                f"on {decision.working_directory or 'no project'}."
            )
            await self.store.update(task)
            await self.activity.update_task(
                task.task_id,
                status="waiting_confirmation",
                worker=decision.worker,
                # A delegation is a decision to approve too, so describe it the
                # same way a tool call is described.
                pending_tool=decision.action or decision.worker,
                pending_arguments={
                    "worker": decision.worker,
                    "project": decision.working_directory or "",
                    **(decision.arguments or {}),
                },
            )
            return self._to_response(task)

        return await self._execute_with_retries(task)

    # ------------------------------------------------------------- assistant

    async def _try_assistant(
        self, task: TaskRecord, *, local_only: bool
    ) -> TaskResponse | None:
        """Give the NPU assistant a shot. ``None`` means "fall through to routing"."""
        settings = load_assistant_settings()
        if not settings.tools.enabled:
            return None

        await self.activity.update_task(task.task_id, status="thinking")

        async def on_step(step) -> None:
            # Record the whole step, not just a one-line summary: the control
            # panel shows the reasoning, the arguments and the result so a run
            # can be followed while it happens rather than explained after.
            await self.activity.record_step(
                task.task_id,
                thought=getattr(step, "thought", "") or "",
                tool=step.tool,
                arguments=dict(getattr(step, "arguments", {}) or {}),
                observation=step.observation or "",
                ok=step.ok,
            )

        try:
            outcome = await self.assistant.run(
                task.message,
                project=task.project,
                local_only=local_only,
                confirmed_tools=set(task.confirmed_tools),
                on_step=on_step,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Assistant loop failed; falling back to worker routing")
            await self.activity.note(
                "notice", "Assistant failed, routing directly", detail=str(exc)[:200],
                task_id=task.task_id,
            )
            return None

        # Nothing happened — let the ordinary worker router handle it.
        if not outcome.steps and not outcome.answer and not outcome.question:
            return None

        return await self._finish_assistant(task, outcome)

    async def _finish_assistant(
        self, task: TaskRecord, outcome: AssistantOutcome
    ) -> TaskResponse:
        # The "needs approval" placeholder is bookkeeping, not work that
        # happened — drop it so a resume does not replay it as history.
        task.assistant_steps = [
            step.model_dump() for step in outcome.steps if step.action != "confirm"
        ]
        task.worker = outcome.delegated[-1] if outcome.delegated else "assistant"

        if outcome.needs_confirmation and outcome.pending_tool:
            task.status = TaskStatus.WAITING_CONFIRMATION
            task.pending_tool = outcome.pending_tool
            task.pending_arguments = outcome.pending_arguments
            task.result = (
                f"Keylane wants to run '{outcome.pending_tool}'"
                + (
                    f" with {outcome.pending_arguments}"
                    if outcome.pending_arguments
                    else ""
                )
                + "."
            )
            await self.store.update(task)
            await self.activity.update_task(
                task.task_id,
                status="waiting_confirmation",
                worker="assistant",
                pending_tool=outcome.pending_tool,
                pending_arguments=dict(outcome.pending_arguments or {}),
            )
            return self._to_response(task)

        task.result = outcome.answer or "Done."
        task.canvas = outcome.canvas
        task.error = outcome.error
        task.evidence = WorkerEvidence(
            worker=task.worker or "assistant",
            action="assist",
            response=task.result,
            stdout="\n\n".join(
                f"{s.tool}: {s.observation}" for s in outcome.steps if s.observation
            )[:8000],
            exit_code=0 if not outcome.error else 1,
        )
        task.status = TaskStatus.FAILED if outcome.error else TaskStatus.COMPLETED
        await self.store.update(task)
        await self.activity.update_task(
            task.task_id,
            status="failed" if outcome.error else "completed",
            worker=task.worker,
            error=outcome.error,
        )
        return self._to_response(task)

    async def _resume_assistant(self, task: TaskRecord) -> TaskResponse:
        """Run the approved tool, then let the assistant carry on from there."""
        from app.assistant import AssistantStep

        pending = task.pending_tool
        arguments = dict(task.pending_arguments)
        if not pending:
            return self._to_response(task)
        if pending not in task.confirmed_tools:
            task.confirmed_tools.append(pending)
        task.pending_tool = None
        task.pending_arguments = {}
        task.status = TaskStatus.RUNNING
        await self.store.update(task)
        await self.activity.update_task(
            task.task_id,
            status="running",
            step=f"approved {pending}",
            clear_pending=True,
        )

        # Replay what already happened so the model does not start over, then
        # execute the step the user just approved.
        prior = [AssistantStep(**step) for step in task.assistant_steps]
        try:
            executed = await self.assistant.run_confirmed_tool(
                pending, arguments, index=len(prior) + 1
            )
            prior.append(executed)

            outcome = await self.assistant.run(
                task.message,
                project=task.project,
                local_only=task.local_only,
                confirmed_tools=set(task.confirmed_tools),
                prior_steps=prior,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Assistant resume failed")
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            await self.store.update(task)
            await self.activity.update_task(
                task.task_id, status="failed", error=str(exc)
            )
            return self._to_response(task)
        return await self._finish_assistant(task, outcome)

    async def _execute_worker(self, decision: RouteDecision) -> WorkerResult:
        return await self.registry.run_worker(decision)

    @staticmethod
    def _worker_canvas(task: TaskRecord, decision: RouteDecision) -> dict[str, Any]:
        """Wrap a worker's output so it renders like an assistant answer."""
        from app.canvas import Block, Link
        from app.canvas_build import markdown_to_canvas

        # Workers answer in prose and markdown, so parse it into blocks rather
        # than showing the raw characters.
        canvas = markdown_to_canvas(
            task.result or "", source=f"via {decision.worker}"
        )
        evidence = task.evidence
        if evidence is not None:
            if evidence.output_path:
                canvas.blocks.append(
                    Block(
                        type="links",
                        links=[Link(label="Output", href=evidence.output_path)],
                    )
                )
            if evidence.changed_files:
                canvas.blocks.append(
                    Block(
                        type="list",
                        entries=evidence.changed_files[:30],
                    )
                )
        return canvas.cleaned().model_dump()

    async def _execute_with_retries(self, task: TaskRecord) -> TaskResponse:
        assert task.route is not None
        decision = task.route
        context_extras: dict[str, Any] = {}
        # Reaching here past a confirmation pause means it was approved.
        await self.activity.update_task(task.task_id, clear_pending=True)

        # Multi-step plan execution
        if task.plan and len(task.plan.steps) > 1:
            return await self._execute_plan(task)

        for attempt in range(task.max_retries + 1):
            if task.task_id in self._cancel:
                task.status = TaskStatus.CANCELLED
                task.error = "Cancelled"
                await self.store.update(task)
                return self._to_response(task)

            task.attempt = attempt
            task.status = TaskStatus.RETRYING if attempt else TaskStatus.RUNNING
            await self.store.update(task)
            await self.activity.update_task(
                task.task_id,
                status=task.status.value,
                worker=decision.worker,
                step=f"attempt {attempt + 1}",
            )

            # Inject verifier next_action / prior context
            run_decision = decision
            if context_extras.get("next_action"):
                run_decision = decision.model_copy(
                    update={
                        "instruction": (
                            f"{decision.instruction}\n\n"
                            f"Previous attempt feedback: {context_extras['next_action']}"
                        )
                    }
                )

            result = await self._execute_worker(run_decision)
            task.evidence = result.evidence
            task.result = result.summary

            task.status = TaskStatus.VERIFYING
            await self.store.update(task)

            verification = self.verifier.verify(
                original_request=task.message,
                task=run_decision,
                evidence=result.evidence,
                attempt=attempt,
            )
            task.verification = verification

            if verification.complete:
                task.status = TaskStatus.COMPLETED
                task.canvas = self._worker_canvas(task, decision)
                await self.store.update(task)
                await self.activity.update_task(
                    task.task_id, status="completed", worker=decision.worker
                )
                return self._to_response(task)

            if not verification.retry or attempt >= task.max_retries:
                task.status = TaskStatus.FAILED
                task.error = verification.reason
                if attempt >= task.max_retries:
                    task.error = f"Maximum retries reached. {verification.reason}"
                await self.store.update(task)
                await self.activity.update_task(
                    task.task_id, status="failed", error=task.error
                )
                return self._to_response(task)

            context_extras["next_action"] = verification.next_action or verification.reason

        task.status = TaskStatus.FAILED
        task.error = "Maximum retries reached."
        await self.store.update(task)
        await self.activity.update_task(
            task.task_id, status="failed", error=task.error
        )
        return self._to_response(task)

    async def _execute_plan(self, task: TaskRecord) -> TaskResponse:
        assert task.plan is not None
        outputs: dict[int, WorkerResult] = {}

        for step in task.plan.steps:
            if task.task_id in self._cancel:
                task.status = TaskStatus.CANCELLED
                await self.store.update(task)
                return self._to_response(task)

            decision = step.decision
            # Pass prior image path into coding step
            if step.depends_on:
                for dep in step.depends_on:
                    prior = outputs.get(dep)
                    if prior and prior.evidence.output_path:
                        decision = decision.model_copy(
                            update={
                                "instruction": (
                                    f"{decision.instruction}\n\n"
                                    f"Generated asset path: {prior.evidence.output_path}"
                                )
                            }
                        )

            try:
                decision = validate_route(
                    decision, local_only=task.local_only, config=self.config
                )
            except PermissionError_ as exc:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                await self.store.update(task)
                return self._to_response(task)

            task.route = decision
            task.worker = decision.worker
            task.status = TaskStatus.RUNNING
            await self.store.update(task)

            # Per-step retry loop
            last_result: WorkerResult | None = None
            for attempt in range(task.max_retries + 1):
                task.attempt = attempt
                run_decision = decision
                if last_result and task.verification and task.verification.next_action:
                    run_decision = decision.model_copy(
                        update={
                            "instruction": (
                                f"{decision.instruction}\n\n"
                                f"Previous attempt feedback: {task.verification.next_action}"
                            )
                        }
                    )
                result = await self._execute_worker(run_decision)
                last_result = result
                task.evidence = result.evidence
                task.result = result.summary
                task.status = TaskStatus.VERIFYING
                await self.store.update(task)

                verification = self.verifier.verify(
                    original_request=task.message,
                    task=run_decision,
                    evidence=result.evidence,
                    attempt=attempt,
                )
                task.verification = verification
                if verification.complete:
                    outputs[step.step] = result
                    break
                if not verification.retry or attempt >= task.max_retries:
                    task.status = TaskStatus.FAILED
                    task.error = verification.reason
                    await self.store.update(task)
                    return self._to_response(task)
                task.status = TaskStatus.RETRYING
                await self.store.update(task)
            else:
                task.status = TaskStatus.FAILED
                task.error = "Step failed after retries."
                await self.store.update(task)
                return self._to_response(task)

        task.status = TaskStatus.COMPLETED
        summaries = [r.summary for r in outputs.values()]
        task.result = "\n\n".join(summaries)
        await self.store.update(task)
        await self.activity.update_task(task.task_id, status="completed")
        return self._to_response(task)

    async def cancel(self, task_id: str, *, reason: str = "Cancelled") -> TaskResponse | None:
        self._cancel.add(task_id)
        task = await self.store.get(task_id)
        if task is None:
            return None
        if task.status not in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            task.status = TaskStatus.CANCELLED
            task.error = reason
            await self.store.update(task)
            await self.activity.update_task(
                task.task_id, status="cancelled", error=reason, clear_pending=True
            )
        return self._to_response(task)

    async def get_task(self, task_id: str) -> TaskResponse | None:
        task = await self.store.get(task_id)
        return self._to_response(task) if task else None
