"""Strict Pydantic models for routing, tasks, and verification."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


ALLOWED_WORKERS = frozenset({"lmstudio", "claude", "cursor", "comfyui", "lemonade"})


def register_worker(worker_id: str) -> None:
    """Allow plugins to register additional worker ids at runtime."""
    global ALLOWED_WORKERS
    ALLOWED_WORKERS = frozenset(set(ALLOWED_WORKERS) | {worker_id.strip().lower()})


def sync_allowed_workers(worker_ids: set[str]) -> None:
    global ALLOWED_WORKERS
    base = {"lmstudio", "claude", "cursor", "comfyui", "lemonade"}
    ALLOWED_WORKERS = frozenset(base | {w.strip().lower() for w in worker_ids})
ALLOWED_INTENTS = frozenset(
    {
        "general_question",
        "coding",
        "image_generation",
        "image_edit",
        "summarization",
        "brainstorming",
        "analysis",
        "multi_step",
    }
)
ALLOWED_ACTIONS = frozenset(
    {
        "answer",
        "summarize",
        "analyze",
        "brainstorm",
        "modify_project",
        "inspect_project",
        "generate_image",
        "edit_image",
        "upscale_image",
        "inpaint_image",
    }
)


class TaskStatus(str, Enum):
    PENDING = "pending"
    ROUTING = "routing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    RUNNING = "running"
    VERIFYING = "verifying"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RouteDecision(BaseModel):
    intent: str
    worker: str
    action: str
    instruction: str
    working_directory: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = False
    workflow: str | None = None
    model: str | None = None

    @field_validator("worker")
    @classmethod
    def worker_must_be_allowed(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ALLOWED_WORKERS:
            raise ValueError(f"worker must be one of {sorted(ALLOWED_WORKERS)}")
        return normalized

    @field_validator("intent")
    @classmethod
    def intent_normalized(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("action")
    @classmethod
    def action_normalized(cls, value: str) -> str:
        return value.strip().lower()


class PlanStep(BaseModel):
    step: int
    decision: RouteDecision
    depends_on: list[int] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    steps: list[PlanStep]
    reason: str = ""


class VerificationResult(BaseModel):
    complete: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    retry: bool = False
    next_action: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    project: str | None = None
    confirmed: bool = False
    local_only: bool | None = None
    task_id: str | None = None


class RouteRequest(BaseModel):
    message: str = Field(min_length=1)
    project: str | None = None
    local_only: bool | None = None


class TranscribeRequest(BaseModel):
    """Audio is uploaded as multipart; this models optional metadata."""

    language: str | None = "en"


class WorkerEvidence(BaseModel):
    worker: str
    action: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    changed_files: list[str] = Field(default_factory=list)
    git_status: str = ""
    git_diff: str = ""
    git_diff_stat: str = ""
    test_exit_code: int | None = None
    build_exit_code: int | None = None
    lint_exit_code: int | None = None
    output_path: str | None = None
    output_dimensions: dict[str, int] | None = None
    response: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerResult(BaseModel):
    success: bool
    evidence: WorkerEvidence
    summary: str = ""
    raw: Any = None


class TaskRecord(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    message: str
    project: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    worker: str | None = None
    route: RouteDecision | None = None
    plan: ExecutionPlan | None = None
    result: str | None = None
    evidence: WorkerEvidence | None = None
    verification: VerificationResult | None = None
    attempt: int = 0
    max_retries: int = 3
    error: str | None = None
    local_only: bool = False
    # Assistant-loop bookkeeping
    assistant_steps: list[dict[str, Any]] = Field(default_factory=list)
    canvas: dict[str, Any] | None = None
    pending_tool: str | None = None
    pending_arguments: dict[str, Any] = Field(default_factory=dict)
    confirmed_tools: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    worker: str | None = None
    result: str | None = None
    route: RouteDecision | None = None
    verification: VerificationResult | None = None
    requires_confirmation: bool = False
    error: str | None = None
    attempt: int = 0
    assistant_steps: list[dict[str, Any]] = Field(default_factory=list)
    canvas: dict[str, Any] | None = None
    """Structured answer, rendered by the popup and the control panel."""

    pending_tool: str | None = None
    pending_arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(BaseModel):
    """Direct tool invocation from the control panel or an integration."""

    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    confirmed: bool = False


class StatusResponse(BaseModel):
    npu: bool
    npu_driver: bool = False
    npu_openvino: bool = False
    npu_detail: str = ""
    openvino_devices: list[str] = Field(default_factory=list)
    lmstudio: bool
    comfyui: bool
    claude: bool
    cursor: bool
    lemonade: bool = False
    gateway: bool = True
    local_only: bool = False
    plugins: dict[str, bool] = Field(default_factory=dict)
    assistant: bool = False
    assistant_device: str | None = None
    assistant_note: str | None = None
    tools_enabled: bool = True
    tool_count: int = 0
    busy: bool = False
    incomplete_models: list["IncompleteModel"] = Field(default_factory=list)
    """Downloads that left a graph behind but no weights.

    Reported separately from "no model" so the panel can offer a resume
    rather than telling you to download what is already half there.
    """


class IncompleteModel(BaseModel):
    id: str
    repo_id: str = ""
    missing: list[str] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    name: str
    path: str


class ProjectsResponse(BaseModel):
    projects: list[ProjectInfo]


class OpenAIMessage(BaseModel):
    role: str
    content: str


class OpenAIChatRequest(BaseModel):
    model: str = "local-agent"
    messages: list[OpenAIMessage]
    stream: bool = False
