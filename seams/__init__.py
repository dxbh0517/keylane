"""Capability seams: interfaces, providers, and the context that composes them."""

from seams.context import Context, build_context, get_context, reset_context
from seams.errors import (
    JobError,
    LlmError,
    SeamError,
    SkillError,
    SubagentError,
    WebError,
)

__all__ = [
    "Context",
    "JobError",
    "LlmError",
    "SeamError",
    "SkillError",
    "SubagentError",
    "WebError",
    "build_context",
    "get_context",
    "reset_context",
]
