"""Pending tool permission approvals."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from daemon.config import permission_mode


@dataclass
class PendingPermission:
    id: str
    tool: str
    arguments: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    resolved: bool = False


_lock = threading.Lock()
_pending: dict[str, PendingPermission] = {}


def create_prompt(tool_name: str, arguments: dict[str, Any]) -> PendingPermission:
    """Register a prompt the interface must answer, whatever the tool's mode.

    Used by `ask_user`, where the prompt *is* the point. Permission modes decide
    whether a dangerous action needs approval; they have nothing to say about a
    question the model is asking.
    """
    pid = str(uuid.uuid4())[:12]
    pending = PendingPermission(id=pid, tool=tool_name, arguments=arguments)
    with _lock:
        _pending[pid] = pending
    return pending


def create_pending(tool_name: str, arguments: dict[str, Any]) -> PendingPermission | None:
    """Register an approval prompt if this tool's mode calls for one."""
    mode = permission_mode(tool_name)
    if mode == "auto":
        return None
    if mode == "deny":
        return PendingPermission(id="", tool=tool_name, arguments=arguments, approved=False, resolved=True)
    return create_prompt(tool_name, arguments)


def wait_pending(pending: PendingPermission, *, timeout: float = 120.0) -> bool:
    if pending.resolved and pending.id == "":
        return pending.approved
    if not pending.event.wait(timeout=timeout):
        with _lock:
            _pending.pop(pending.id, None)
        return False
    with _lock:
        _pending.pop(pending.id, None)
    return pending.approved


def get_pending() -> list[dict[str, Any]]:
    with _lock:
        return [
            {"id": p.id, "tool": p.tool, "arguments": p.arguments}
            for p in _pending.values()
            if not p.resolved
        ]


def respond(permission_id: str, approved: bool) -> bool:
    with _lock:
        pending = _pending.get(permission_id)
        if not pending or pending.resolved:
            return False
        pending.approved = approved
        pending.resolved = True
    pending.event.set()
    return True
