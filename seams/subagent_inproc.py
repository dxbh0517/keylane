"""The in-process subagent provider.

A child is an ordinary `AIAgent` on a fresh session with a restricted view of
the parent's tool registry. It is not a fork: it sees none of the parent's
conversation, which is exactly why delegating keeps the parent's context small.
"""

from __future__ import annotations

import asyncio
import logging

from seams.subagent import (
    SubagentCapabilities,
    SubagentRequest,
    SubagentResult,
)

logger = logging.getLogger(__name__)


class InProcessSubagentProvider:
    """Runs the child on this machine, through the configured model route."""

    capabilities = SubagentCapabilities(tool_filter=True, route_choice=True, persona=True)

    def __init__(self, provider_id: str = "local") -> None:
        self.id = provider_id

    def available(self) -> bool:
        from seams import get_context

        # Ready if any route can serve the child. The route it will actually
        # use is resolved per call, so a GPU model coming online mid-session is
        # picked up without re-registering anything.
        return get_context().llm.is_ready("background") or get_context().llm.is_ready(
            "interactive"
        )

    def start(self, request: SubagentRequest) -> SubagentResult:
        from agent.loop import AIAgent
        from tools.registry import get_registry

        # The child is built around its scope rather than replacing one it
        # already made, and it runs on the requested route — which is why a
        # configured GPU model takes the delegated work and leaves the NPU free.
        agent = AIAgent(route=request.route)
        scope = get_registry().child(f"subagent:{agent.session_id}")
        try:
            scope.restrict(
                allow=list(request.tool_allow) if request.tool_allow is not None else None,
                deny=[name for name in request.tool_deny if get_registry().get(name)],
            )
        except ValueError as exc:
            return SubagentResult(
                output="",
                stop_reason="error",
                diagnostic=f"could not apply the tool filter: {exc}",
                session_id=agent.session_id,
            )
        agent.tools = scope

        if request.cancel is not None and request.cancel.is_set():
            return SubagentResult(
                output="", stop_reason="aborted", session_id=agent.session_id
            )

        try:
            result = asyncio.run(agent.run(request.prompt))
        except Exception as exc:  # noqa: BLE001
            logger.exception("subagent run failed")
            return SubagentResult(
                output="",
                stop_reason="error",
                diagnostic=str(exc)[:4096],
                session_id=agent.session_id,
            )

        if request.cancel is not None and request.cancel.is_set():
            return SubagentResult(
                output=result.answer,
                stop_reason="aborted",
                session_id=agent.session_id,
            )
        return SubagentResult(output=result.answer, session_id=agent.session_id)
