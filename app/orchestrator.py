"""Task orchestration: route → confirm → execute → verify → retry."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import AppConfig, get_config
from app.permissions import PermissionError_, validate_route
from app.planner import build_plan, needs_multi_step
from app.plugins.registry import get_plugin_registry
from app.router import RouterService
from app.schemas import (
    ChatRequest,
    RouteDecision,
    TaskRecord,
    TaskResponse,
    TaskStatus,
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
            requires_confirmation=bool(
                task.route and task.route.requires_confirmation
                and task.status == TaskStatus.WAITING_CONFIRMATION
            ),
            error=task.error,
            attempt=task.attempt,
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

    async def chat(self, request: ChatRequest) -> TaskResponse:
        # Resume confirmation
        if request.task_id and request.confirmed:
            task = await self.store.get(request.task_id)
            if task is None:
                return TaskResponse(
                    task_id=request.task_id,
                    status=TaskStatus.FAILED,
                    error="Unknown task_id",
                )
            if task.status != TaskStatus.WAITING_CONFIRMATION:
                return self._to_response(task)
            return await self._execute_with_retries(task)

        task = TaskRecord(
            message=request.message,
            project=request.project,
            max_retries=self.config.gateway.max_retries,
            status=TaskStatus.ROUTING,
        )
        await self.store.put(task)

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
            else:
                decision = await self.router.route(
                    request.message,
                    project=request.project,
                    local_only=request.local_only,
                )
                task.plan = None

            task.route = decision
            task.worker = decision.worker
        except PermissionError_ as exc:
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            await self.store.update(task)
            return self._to_response(task)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Routing failed")
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            await self.store.update(task)
            return self._to_response(task)

        if decision.requires_confirmation and not request.confirmed:
            task.status = TaskStatus.WAITING_CONFIRMATION
            task.result = (
                f"{decision.worker} wants to run '{decision.action}' "
                f"on {decision.working_directory or 'no project'}."
            )
            await self.store.update(task)
            return self._to_response(task)

        return await self._execute_with_retries(task)

    async def _execute_worker(self, decision: RouteDecision) -> WorkerResult:
        return await self.registry.run_worker(decision)

    async def _execute_with_retries(self, task: TaskRecord) -> TaskResponse:
        assert task.route is not None
        decision = task.route
        context_extras: dict[str, Any] = {}

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
                await self.store.update(task)
                return self._to_response(task)

            if not verification.retry or attempt >= task.max_retries:
                task.status = TaskStatus.FAILED
                task.error = verification.reason
                if attempt >= task.max_retries:
                    task.error = f"Maximum retries reached. {verification.reason}"
                await self.store.update(task)
                return self._to_response(task)

            context_extras["next_action"] = verification.next_action or verification.reason

        task.status = TaskStatus.FAILED
        task.error = "Maximum retries reached."
        await self.store.update(task)
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
                decision = validate_route(decision, config=self.config)
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
        return self._to_response(task)

    async def cancel(self, task_id: str) -> TaskResponse | None:
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
            task.error = "Cancelled"
            await self.store.update(task)
        return self._to_response(task)

    async def get_task(self, task_id: str) -> TaskResponse | None:
        task = await self.store.get(task_id)
        return self._to_response(task) if task else None
