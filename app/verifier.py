"""Verification service — NPU judges; gateway owns control flow."""

from __future__ import annotations

from app.config import AppConfig, get_config
from app.npu.verifier_model import get_verifier_model
from app.schemas import RouteDecision, VerificationResult, WorkerEvidence


class VerifierService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.model = get_verifier_model(self.config)

    def verify(
        self,
        *,
        original_request: str,
        task: RouteDecision,
        evidence: WorkerEvidence,
        attempt: int = 0,
    ) -> VerificationResult:
        return self.model.verify(
            original_request=original_request,
            task=task,
            evidence=evidence,
            attempt=attempt,
        )
