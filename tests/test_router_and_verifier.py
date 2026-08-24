"""Unit tests for routing, permissions, and verification heuristics.

These exercise the fallback paths directly, so the models are built with
``__new__`` to skip loading an OpenVINO pipeline.
"""

from __future__ import annotations

import pytest

from app.npu.router_model import RouterModel
from app.npu.verifier_model import VerifierModel
from app.permissions import PermissionError_, enforce_local_only, validate_route
from app.schemas import RouteDecision, WorkerEvidence


def test_heuristic_routes_coding_to_cursor_or_claude():
    model = RouterModel.__new__(RouterModel)
    decision = model._heuristic_route(
        "Fix the authentication bug in my project",
        project="/home/emul/Documents/Code/aurora",
        local_only=False,
        available_workers={"cursor", "claude", "lmstudio", "comfyui"},
    )
    assert decision.worker in {"cursor", "claude"}
    assert decision.action == "modify_project"
    assert decision.requires_confirmation is True


def test_heuristic_routes_image_to_comfyui():
    model = RouterModel.__new__(RouterModel)
    decision = model._heuristic_route(
        "Generate a 1536x1024 cyberpunk city background",
        project=None,
        local_only=False,
        available_workers={"lmstudio", "comfyui"},
    )
    assert decision.worker == "comfyui"
    assert decision.workflow == "flux_txt2img"


def test_heuristic_respects_local_only():
    model = RouterModel.__new__(RouterModel)
    decision = model._heuristic_route(
        "Use Claude to fix this",
        project="/home/emul/Documents/Code/aurora",
        local_only=True,
        available_workers={"lmstudio", "comfyui"},
    )
    assert decision.worker in {"lmstudio", "comfyui"}


def test_local_only_blocks_cloud_workers():
    with pytest.raises(PermissionError_):
        enforce_local_only("claude", True)
    with pytest.raises(PermissionError_):
        enforce_local_only("cursor", True)
    enforce_local_only("lmstudio", True)


def test_validate_route_rejects_shell_smuggling():
    decision = RouteDecision(
        intent="coding",
        worker="cursor",
        action="modify_project",
        instruction="fix it",
        working_directory="/home/emul/Documents/Code/aurora",
        arguments={"command": "rm -rf /"},
        requires_confirmation=True,
    )
    with pytest.raises(PermissionError_):
        validate_route(decision)


def test_validate_route_rejects_path_outside_roots():
    decision = RouteDecision(
        intent="coding",
        worker="cursor",
        action="modify_project",
        instruction="fix it",
        working_directory="/etc",
        requires_confirmation=True,
    )
    with pytest.raises(PermissionError_):
        validate_route(decision)


def test_verifier_detects_build_failure():
    verifier = VerifierModel.__new__(VerifierModel)
    evidence = WorkerEvidence(
        worker="cursor",
        action="modify_project",
        exit_code=0,
        build_exit_code=1,
        stderr="TS2345",
        changed_files=["src/auth.ts"],
    )
    task = RouteDecision(
        intent="coding",
        worker="cursor",
        action="modify_project",
        instruction="Fix auth",
        working_directory="/home/emul/Documents/Code/aurora",
    )
    result = verifier._heuristic_verify(
        original_request="Fix auth",
        task=task,
        evidence=evidence,
    )
    assert result.complete is False
    assert result.retry is True


def test_verifier_accepts_image_output(tmp_path):
    verifier = VerifierModel.__new__(VerifierModel)
    out = tmp_path / "out.png"
    out.write_bytes(b"fake")
    evidence = WorkerEvidence(
        worker="comfyui",
        action="generate_image",
        exit_code=0,
        output_path=str(out),
        output_dimensions={"width": 1536, "height": 1024},
    )
    task = RouteDecision(
        intent="image_generation",
        worker="comfyui",
        action="generate_image",
        instruction="city",
        arguments={"width": 1536, "height": 1024},
        workflow="flux_txt2img",
    )
    result = verifier._heuristic_verify(
        original_request="Generate city",
        task=task,
        evidence=evidence,
    )
    assert result.complete is True
