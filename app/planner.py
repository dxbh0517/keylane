"""Multi-step planning for requests that need more than one worker."""

from __future__ import annotations

import re

from app.schemas import ExecutionPlan, PlanStep, RouteDecision


def needs_multi_step(message: str) -> bool:
    text = message.lower()
    has_image = any(
        tok in text
        for tok in ("image", "artwork", "hero", "background", "generate a", "create an image")
    )
    has_build = any(
        tok in text for tok in ("build", "landing page", "website", "integrate", "project")
    )
    return has_image and has_build


def build_plan(message: str, project: str | None) -> ExecutionPlan:
    """Deterministic planner for the MVP multi-worker path."""
    prompt_match = re.search(
        r"(?:generate|create|make)\s+(?:an?\s+)?(.+?)(?:\s+for\s+|\s+and\s+|$)",
        message,
        flags=re.IGNORECASE,
    )
    image_prompt = prompt_match.group(1).strip() if prompt_match else message

    step1 = PlanStep(
        step=1,
        decision=RouteDecision(
            intent="image_generation",
            worker="comfyui",
            action="generate_image",
            instruction=f"Generate: {image_prompt}",
            working_directory=project,
            workflow="flux_txt2img",
            arguments={"prompt": image_prompt, "width": 1536, "height": 1024},
            requires_confirmation=False,
        ),
        depends_on=[],
    )
    step2 = PlanStep(
        step=2,
        decision=RouteDecision(
            intent="coding",
            worker="cursor",
            action="modify_project",
            instruction=(
                f"{message}\n\n"
                "Integrate the generated hero image from the previous step "
                "into the project. Image path will be provided as context."
            ),
            working_directory=project,
            requires_confirmation=True,
        ),
        depends_on=[1],
    )
    return ExecutionPlan(
        steps=[step1, step2],
        reason="Image generation followed by project integration.",
    )
