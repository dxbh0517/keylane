"""Project sandbox and confirmation policy."""

from __future__ import annotations

from pathlib import Path

from app.config import AppConfig, get_config
from app.schemas import RouteDecision


CLOUD_WORKERS = frozenset({"claude", "cursor"})
MODIFY_ACTIONS = frozenset(
    {
        "modify_project",
        "edit_image",
        "inpaint_image",
    }
)
DESTRUCTIVE_KEYWORDS = (
    "delete",
    "remove",
    "rm ",
    "destroy",
    "drop database",
    "format disk",
)


class PermissionError_(Exception):
    """Raised when a route violates security policy."""


def resolve_under_roots(path: str | Path, roots: list[str]) -> Path:
    resolved = Path(path).expanduser().resolve()
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        try:
            resolved.relative_to(root_path)
            return resolved
        except ValueError:
            continue
    raise PermissionError_(
        f"Path '{resolved}' is outside allowed project roots: {roots}"
    )


def validate_working_directory(
    path: str | None,
    config: AppConfig | None = None,
    *,
    required: bool = False,
) -> str | None:
    cfg = config or get_config()
    if path is None or path.strip() == "":
        if required:
            raise PermissionError_(
                "A project directory is required for this worker. Pick one "
                "from the project chip in the popup, or add one under "
                "Projects in the control panel."
            )
        return None
    return str(resolve_under_roots(path, cfg.security.allowed_project_roots))


def is_local_only(config: AppConfig | None = None, override: bool | None = None) -> bool:
    cfg = config or get_config()
    if override is not None:
        return override
    return cfg.gateway.local_only


def enforce_local_only(worker: str, local_only: bool) -> None:
    if local_only and worker in CLOUD_WORKERS:
        raise PermissionError_(
            f"Worker '{worker}' is blocked in local-only mode. "
            "Allowed: lmstudio, comfyui (and NPU)."
        )


def looks_destructive(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in DESTRUCTIVE_KEYWORDS)


def apply_confirmation_policy(
    decision: RouteDecision,
    config: AppConfig | None = None,
) -> RouteDecision:
    cfg = config or get_config()
    data = decision.model_dump()

    if decision.action in MODIFY_ACTIONS and cfg.security.require_confirmation_for_modifications:
        data["requires_confirmation"] = True

    if looks_destructive(decision.instruction) or looks_destructive(
        str(decision.arguments)
    ):
        data["requires_confirmation"] = True

    return RouteDecision(**data)


def validate_route(
    decision: RouteDecision,
    *,
    local_only: bool = False,
    available_workers: set[str] | None = None,
    config: AppConfig | None = None,
) -> RouteDecision:
    cfg = config or get_config()
    enforce_local_only(decision.worker, local_only)

    if available_workers is not None and decision.worker not in available_workers:
        raise PermissionError_(
            f"Worker '{decision.worker}' is unavailable. "
            f"Available: {sorted(available_workers)}"
        )

    coding_workers = {"claude", "cursor"}
    working_directory = validate_working_directory(
        decision.working_directory,
        cfg,
        required=decision.worker in coding_workers,
    )
    data = decision.model_dump()
    data["working_directory"] = working_directory

    # Reject any attempt to smuggle shell commands through arguments.
    for key in ("command", "shell", "bash", "cmd"):
        if key in data.get("arguments", {}):
            raise PermissionError_(
                f"Argument '{key}' is not allowed. Workers accept structured actions only."
            )

    validated = RouteDecision(**data)
    return apply_confirmation_policy(validated, cfg)
