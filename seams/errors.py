"""Structured seam errors.

A tool that fails with a bare string leaves the model guessing. Every seam
failure carries a stable ``code`` the caller can branch on and a message
written to be read by the model, so it can pick a different action instead of
retrying the same one.

``code`` is deliberately open, not an enum: a provider may raise its own code,
and consumers must tolerate one they do not recognise.
"""

from __future__ import annotations

from typing import Any


class SeamError(Exception):
    """A capability could not run. The message is model-facing."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"error": self.message, "code": self.code}
        payload.update(self.detail)
        return payload

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.code}] {self.message}"


class LlmError(SeamError):
    """The model routing seam could not serve a request."""


class WebError(SeamError):
    """The web seam could not search or fetch."""


class SkillError(SeamError):
    """The skill seam could not list or load."""


class JobError(SeamError):
    """The background-job seam could not start, read, or stop work."""


class SubagentError(SeamError):
    """The delegation seam could not start or reach a child."""
