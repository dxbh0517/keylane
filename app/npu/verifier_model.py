"""OpenVINO NPU verifier model with evidence-based fallback."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.config import AppConfig, get_config
from app.schemas import RouteDecision, VerificationResult, WorkerEvidence

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM_PROMPT = """You are the verification model for a local Fedora AI gateway.

You do not execute commands or modify files.
Your only job is to decide whether the worker actually fulfilled the user request.

Return ONLY valid JSON:
{
  "complete": true|false,
  "confidence": 0.0-1.0,
  "reason": "...",
  "retry": true|false,
  "next_action": "..." or null
}

Rules:
1. Be strict about build/test failures for coding tasks.
2. For images, require an output path and matching dimensions when requested.
3. Never invent evidence that is not provided.
4. If incomplete but fixable, set retry=true and give a concrete next_action.
"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in verifier output: {text[:200]}")
    return json.loads(text[start : end + 1])


def _sanitize_evidence(evidence: WorkerEvidence) -> dict[str, Any]:
    """Strip secrets-like content before sending to the verifier."""
    data = evidence.model_dump()
    for key in ("stdout", "stderr", "git_diff", "response", "git_status"):
        value = data.get(key) or ""
        # Redact common secret patterns without exposing values.
        value = re.sub(
            r"(?i)(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+",
            r"\1=[REDACTED]",
            value,
        )
        value = re.sub(r"(?i)\.env\b.*", "[REDACTED .env reference]", value)
        if len(value) > 8000:
            value = value[:8000] + "\n...[truncated]"
        data[key] = value
    return data


class VerifierModel:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._pipeline = None
        self._device: str | None = None
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        # Reuse the same small model directory for the first prototype.
        from app.npu.router_model import RouterModel

        model_path = self.config.npu_model_path
        if not RouterModel._model_ready(model_path):
            logger.warning("Verifier model missing — using heuristic verifier.")
            return
        try:
            import openvino as ov
            import openvino_genai as ov_genai

            core = ov.Core()
            devices = list(core.available_devices)
            preferred = self.config.npu.device
            fallback = self.config.npu.fallback_device
            device = preferred if preferred in devices else (
                fallback if fallback in devices else None
            )
            if device is None:
                return
            self._pipeline = ov_genai.LLMPipeline(str(model_path), device)
            self._device = device
            logger.info("Verifier model loaded on %s", device)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load verifier model: %s", exc)

    def verify(
        self,
        *,
        original_request: str,
        task: RouteDecision,
        evidence: WorkerEvidence,
        attempt: int = 0,
    ) -> VerificationResult:
        if self._pipeline is not None:
            try:
                return self._verify_with_model(
                    original_request=original_request,
                    task=task,
                    evidence=evidence,
                    attempt=attempt,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("NPU verifier failed (%s); using heuristic.", exc)
        return self._heuristic_verify(
            original_request=original_request,
            task=task,
            evidence=evidence,
        )

    def _verify_with_model(
        self,
        *,
        original_request: str,
        task: RouteDecision,
        evidence: WorkerEvidence,
        attempt: int,
    ) -> VerificationResult:
        assert self._pipeline is not None
        payload = {
            "original_request": original_request,
            "task": task.model_dump(),
            "evidence": _sanitize_evidence(evidence),
            "attempt": attempt,
        }
        prompt = f"{VERIFIER_SYSTEM_PROMPT}\n\nINPUT:\n{json.dumps(payload, indent=2)}\n"
        raw = self._pipeline.generate(prompt, max_new_tokens=256)
        text = raw.texts[0] if hasattr(raw, "texts") and raw.texts else str(raw)
        data = _extract_json(text)
        return VerificationResult(**data)

    def _heuristic_verify(
        self,
        *,
        original_request: str,
        task: RouteDecision,
        evidence: WorkerEvidence,
    ) -> VerificationResult:
        if task.worker in {"claude", "cursor"}:
            if evidence.exit_code not in (0, None):
                return VerificationResult(
                    complete=False,
                    confidence=0.9,
                    reason=f"Worker exited with code {evidence.exit_code}.",
                    retry=True,
                    next_action=(
                        f"Address the error below and retry.\n{evidence.stderr[-2000:]}"
                        if evidence.stderr
                        else "Retry the coding task and ensure the command succeeds."
                    ),
                )
            if evidence.build_exit_code not in (0, None):
                return VerificationResult(
                    complete=False,
                    confidence=0.95,
                    reason="The project build failed.",
                    retry=True,
                    next_action=(
                        "Fix the build errors and rerun the build.\n"
                        + (evidence.stderr[-2000:] if evidence.stderr else "")
                    ),
                )
            if evidence.test_exit_code not in (0, None):
                return VerificationResult(
                    complete=False,
                    confidence=0.9,
                    reason="Tests failed after the worker finished.",
                    retry=True,
                    next_action="Fix failing tests and rerun them.",
                )
            if task.action == "modify_project" and not evidence.changed_files and not evidence.git_diff:
                return VerificationResult(
                    complete=False,
                    confidence=0.7,
                    reason="No file changes were detected for a modification request.",
                    retry=True,
                    next_action="Make the required code changes and verify with git diff.",
                )
            return VerificationResult(
                complete=True,
                confidence=0.85,
                reason="Worker succeeded with acceptable exit/build/test signals.",
                retry=False,
                next_action=None,
            )

        if task.worker == "comfyui":
            if not evidence.output_path:
                return VerificationResult(
                    complete=False,
                    confidence=0.95,
                    reason="No image output path was produced.",
                    retry=True,
                    next_action="Regenerate the image and ensure an output file is written.",
                )
            from pathlib import Path

            if not Path(evidence.output_path).exists():
                return VerificationResult(
                    complete=False,
                    confidence=0.98,
                    reason=f"Output file missing: {evidence.output_path}",
                    retry=True,
                    next_action="Regenerate and confirm the output file exists.",
                )
            requested_w = task.arguments.get("width")
            requested_h = task.arguments.get("height")
            dims = evidence.output_dimensions or {}
            if requested_w and dims.get("width") and int(dims["width"]) != int(requested_w):
                return VerificationResult(
                    complete=False,
                    confidence=0.9,
                    reason=f"Width mismatch: wanted {requested_w}, got {dims.get('width')}.",
                    retry=True,
                    next_action=f"Regenerate at width={requested_w}, height={requested_h}.",
                )
            if requested_h and dims.get("height") and int(dims["height"]) != int(requested_h):
                return VerificationResult(
                    complete=False,
                    confidence=0.9,
                    reason=f"Height mismatch: wanted {requested_h}, got {dims.get('height')}.",
                    retry=True,
                    next_action=f"Regenerate at width={requested_w}, height={requested_h}.",
                )
            return VerificationResult(
                complete=True,
                confidence=0.92,
                reason="Image output exists and matches requested parameters where specified.",
                retry=False,
                next_action=None,
            )

        # lmstudio / default
        if not (evidence.response or evidence.stdout):
            return VerificationResult(
                complete=False,
                confidence=0.8,
                reason="Empty response from LM Studio worker.",
                retry=True,
                next_action="Retry the question and return a non-empty answer.",
            )
        return VerificationResult(
            complete=True,
            confidence=0.8,
            reason="Received a non-empty local model response.",
            retry=False,
            next_action=None,
        )


_verifier: VerifierModel | None = None


def get_verifier_model(config: AppConfig | None = None) -> VerifierModel:
    global _verifier
    if _verifier is None:
        _verifier = VerifierModel(config)
    return _verifier
