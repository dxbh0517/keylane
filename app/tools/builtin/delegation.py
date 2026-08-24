"""Delegation tools — hand a task to a configured AI worker and follow up.

These are what turn the small NPU model into a *supervisor*: it tries a job with
its own tools first, and when the job needs a real coding agent or an image
model it delegates, then checks the returned evidence against the original ask.
"""

from __future__ import annotations

import logging
from typing import Any

from app.assistant_settings import load_assistant_settings
from app.config import AppConfig, get_config
from app.permissions import PermissionError_, validate_route
from app.plugins.registry import PluginRegistry, get_plugin_registry
from app.schemas import RouteDecision
from app.tools.base import (
    BaseTool,
    ToolDanger,
    ToolResult,
    object_schema,
    string_prop,
)

logger = logging.getLogger(__name__)

# What each known worker is good at, shown to the assistant so it picks well.
WORKER_HINTS = {
    "claude": "Claude Code — multi-file repository work, refactors, tests, deep reasoning about code. Needs a project.",
    "cursor": "Cursor CLI — repository edits and code generation. Needs a project.",
    "comfyui": "ComfyUI — image generation, editing, inpainting and upscaling.",
    "lmstudio": "LM Studio — a larger local chat model for long-form writing, analysis and summarising.",
    "lemonade": "Lemonade Server — a second local chat model, OpenAI-compatible.",
}

ACTION_BY_INTENT = {
    "coding": "modify_project",
    "image_generation": "generate_image",
    "image_edit": "edit_image",
    "analysis": "analyze",
    "summarization": "summarize",
    "brainstorming": "brainstorm",
    "general_question": "answer",
}


class ListWorkersTool(BaseTool):
    name = "list_workers"
    description = (
        "List the AI workers that are configured, enabled and healthy right now, "
        "with what each is good at. Call this before delegating if unsure."
    )
    danger = ToolDanger.SAFE
    category = "delegation"

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config

    @property
    def registry(self) -> PluginRegistry:
        return self._registry or get_plugin_registry(self._config)

    async def run(self, args: dict[str, Any]) -> ToolResult:
        local_only = bool((self._config or get_config()).gateway.local_only)
        try:
            available = await self.registry.available_workers(local_only=local_only)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Could not query workers: {exc}")
        enabled = self.registry.enabled_worker_ids(local_only=local_only)

        rows = []
        for worker in sorted(enabled):
            hint = WORKER_HINTS.get(worker, "Plugin-provided worker.")
            state = "ready" if worker in available else "configured but not reachable"
            rows.append(f"- {worker} [{state}] — {hint}")

        if not rows:
            return ToolResult.success(
                "No AI workers are enabled.", data={"workers": [], "available": []}
            )
        return ToolResult.success(
            "\n".join(rows),
            data={"workers": sorted(enabled), "available": sorted(available)},
        )


class DelegateTool(BaseTool):
    name = "delegate_to_worker"
    description = (
        "Hand a task you cannot do yourself to a more capable configured AI worker "
        "(Claude Code, Cursor, ComfyUI, LM Studio…). Give a complete, self-contained "
        "instruction — the worker cannot see this conversation. Returns the worker's "
        "result and evidence so you can check it."
    )
    danger = ToolDanger.SENSITIVE
    category = "delegation"

    def __init__(
        self,
        registry: PluginRegistry | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self._registry = registry
        self._config = config

    @property
    def registry(self) -> PluginRegistry:
        return self._registry or get_plugin_registry(self._config)

    @property
    def config(self) -> AppConfig:
        return self._config or get_config()

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "worker": string_prop(
                    "Worker id, e.g. claude, cursor, comfyui, lmstudio. "
                    "Call list_workers if unsure."
                ),
                "instruction": string_prop(
                    "The full, self-contained task for the worker, including all "
                    "context it needs."
                ),
                "intent": string_prop(
                    "What kind of work this is.",
                    enum=sorted(ACTION_BY_INTENT),
                    default="general_question",
                ),
                "project": string_prop(
                    "Absolute path of the project directory — required for coding workers."
                ),
                "reason": string_prop("Why you are delegating instead of doing it yourself."),
            },
            required=["worker", "instruction"],
        )

    def availability(self) -> str | None:
        if not load_assistant_settings().delegation.enabled:
            return "Delegation is disabled in assistant settings"
        return None

    async def run(self, args: dict[str, Any]) -> ToolResult:
        settings = load_assistant_settings().delegation
        if not settings.enabled:
            return ToolResult.failure("Delegation is disabled in assistant settings.")

        worker = str(args.get("worker") or "").strip().lower()
        instruction = str(args.get("instruction") or "").strip()
        if not worker:
            return ToolResult.failure("No worker given.")
        if not instruction:
            return ToolResult.failure("An instruction is required.")

        local_only = bool(self.config.gateway.local_only)
        enabled = self.registry.enabled_worker_ids(local_only=local_only)
        if worker not in enabled:
            return ToolResult.failure(
                f"Worker '{worker}' is not available"
                + (" in local-only mode" if local_only else "")
                + f". Enabled workers: {', '.join(sorted(enabled)) or 'none'}."
            )

        intent = str(args.get("intent") or "general_question").strip().lower()
        action = ACTION_BY_INTENT.get(intent, "answer")
        project = str(args.get("project") or "").strip() or None

        decision = RouteDecision(
            intent=intent if intent in ACTION_BY_INTENT else "general_question",
            worker=worker,
            action=action,
            instruction=instruction,
            working_directory=project,
            arguments={"prompt": instruction} if worker == "comfyui" else {},
            requires_confirmation=False,
        )

        try:
            # The gateway — not the model — decides what is legal here.
            decision = validate_route(
                decision, local_only=local_only, config=self.config
            )
        except PermissionError_ as exc:
            return ToolResult.failure(str(exc))

        try:
            result = await self.registry.run_worker(decision)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Delegation to %s failed", worker)
            return ToolResult.failure(f"{worker} failed: {exc}")

        evidence = result.evidence
        detail = {
            "worker": worker,
            "action": decision.action,
            "success": result.success,
            "exit_code": evidence.exit_code,
            "changed_files": evidence.changed_files,
            "output_path": evidence.output_path,
            "stderr": (evidence.stderr or "")[:1500],
        }
        summary = (result.summary or evidence.response or "").strip()
        if not summary:
            summary = f"{worker} finished with exit code {evidence.exit_code}."

        body = (
            f"[{worker} → {decision.action}] "
            f"{'succeeded' if result.success else 'FAILED'}\n\n{summary[:6000]}"
        )
        if evidence.changed_files:
            body += "\n\nChanged files:\n" + "\n".join(evidence.changed_files[:40])
        if evidence.output_path:
            body += f"\n\nOutput file: {evidence.output_path}"

        return ToolResult(
            ok=result.success,
            output=body,
            error=None if result.success else (evidence.stderr or "worker failed")[:500],
            data=detail,
            artifacts=[evidence.output_path] if evidence.output_path else [],
        )


class VerifyDelegatedResultTool(BaseTool):
    name = "verify_result"
    description = (
        "Ask the verifier to judge whether the work done so far actually satisfies "
        "the user's original request. Use this after delegating, before you report "
        "back to the user."
    )
    danger = ToolDanger.SAFE
    category = "delegation"

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config

    def parameters(self) -> dict[str, Any]:
        return object_schema(
            {
                "original_request": string_prop("What the user actually asked for."),
                "work_done": string_prop("A factual description of what was produced."),
            },
            required=["original_request", "work_done"],
        )

    async def run(self, args: dict[str, Any]) -> ToolResult:
        from app.npu.verifier_model import get_verifier_model
        from app.schemas import WorkerEvidence

        original = str(args.get("original_request") or "").strip()
        work = str(args.get("work_done") or "").strip()
        if not original or not work:
            return ToolResult.failure("Both original_request and work_done are required.")

        decision = RouteDecision(
            intent="analysis",
            worker="lmstudio",
            action="analyze",
            instruction=original,
        )
        evidence = WorkerEvidence(
            worker="assistant", action="analyze", response=work, stdout=work, exit_code=0
        )
        verification = get_verifier_model(self._config).verify(
            original_request=original, task=decision, evidence=evidence, attempt=0
        )
        verdict = "COMPLETE" if verification.complete else "INCOMPLETE"
        body = (
            f"{verdict} (confidence {verification.confidence:.2f})\n{verification.reason}"
        )
        if verification.next_action:
            body += f"\nSuggested next step: {verification.next_action}"
        return ToolResult(
            ok=verification.complete,
            output=body,
            data=verification.model_dump(),
        )


def delegation_tools(
    registry: PluginRegistry | None = None,
    config: AppConfig | None = None,
) -> list[BaseTool]:
    return [
        ListWorkersTool(registry, config),
        DelegateTool(registry, config),
        VerifyDelegatedResultTool(config),
    ]
