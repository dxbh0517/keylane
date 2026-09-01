"""Delegation: capability checks before start, depth caps, and tool filters."""

from __future__ import annotations

import pytest

from seams.errors import SubagentError
from seams.subagent import (
    DEFAULT_CHILD_DENY,
    SubagentCapabilities,
    SubagentRequest,
    SubagentResult,
    SubagentRuntime,
)


class FakeProvider:
    def __init__(self, provider_id="local", ready=True, **caps):
        self.id = provider_id
        self.ready = ready
        self.capabilities = SubagentCapabilities(**caps)
        self.requests: list[SubagentRequest] = []

    def available(self) -> bool:
        return self.ready

    def start(self, request: SubagentRequest) -> SubagentResult:
        self.requests.append(request)
        return SubagentResult(output=f"did: {request.prompt}")


@pytest.fixture()
def runtime():
    return SubagentRuntime()


def _request(**kwargs) -> SubagentRequest:
    kwargs.setdefault("prompt", "Research OpenVINO NPU support")
    kwargs.setdefault("tool_deny", ())
    return SubagentRequest(**kwargs)


def test_a_delegation_returns_the_childs_result(runtime) -> None:
    runtime.register(FakeProvider())
    result = runtime.start(_request())
    assert result.output == "did: Research OpenVINO NPU support"
    assert result.stop_reason == "completed"


def test_an_unsupported_capability_fails_before_the_child_exists(runtime) -> None:
    """Accept-then-ignore would give a child every tool while claiming not to."""
    provider = FakeProvider(tool_filter=False)
    runtime.register(provider)
    with pytest.raises(SubagentError) as exc:
        runtime.start(_request(tool_deny=("shell",)))
    assert exc.value.code == "SUBAGENT_UNSUPPORTED_CAPABILITY"
    assert provider.requests == []


def test_a_persona_needs_the_matching_capability(runtime) -> None:
    runtime.register(FakeProvider(tool_filter=True, persona=False))
    with pytest.raises(SubagentError, match="persona"):
        runtime.start(_request(persona="Be terse."))


def test_a_supported_capability_is_passed_through(runtime) -> None:
    provider = FakeProvider(tool_filter=True)
    runtime.register(provider)
    runtime.start(_request(tool_deny=("shell",)))
    assert provider.requests[0].tool_deny == ("shell",)


def test_an_empty_prompt_is_refused(runtime) -> None:
    """A child cannot see this conversation, so an empty prompt says nothing."""
    runtime.register(FakeProvider())
    with pytest.raises(SubagentError) as exc:
        runtime.start(_request(prompt="   "))
    assert exc.value.code == "SUBAGENT_INVALID_REQUEST"
    assert "self-contained" in exc.value.message


def test_no_ready_provider_is_a_structured_failure(runtime) -> None:
    runtime.register(FakeProvider(ready=False))
    with pytest.raises(SubagentError) as exc:
        runtime.start(_request())
    assert exc.value.code == "SUBAGENT_UNAVAILABLE"


def test_a_named_provider_that_is_missing_fails_loudly(runtime) -> None:
    runtime.register(FakeProvider())
    with pytest.raises(SubagentError) as exc:
        runtime.start(_request(), provider_id="cloud")
    assert exc.value.code == "SUBAGENT_PROVIDER_MISSING"


def test_duplicate_provider_ids_are_a_programming_error(runtime) -> None:
    runtime.register(FakeProvider())
    with pytest.raises(SubagentError, match="already registered"):
        runtime.register(FakeProvider())


def test_selection_follows_the_declared_preference(runtime) -> None:
    runtime.register(FakeProvider("local"))
    runtime.register(FakeProvider("cloud"))
    runtime.configure_preference(["cloud", "local"])
    assert runtime.resolve().id == "cloud"


def test_delegation_cannot_nest_without_end(runtime, monkeypatch) -> None:
    from seams import jobs

    runtime.register(FakeProvider())
    monkeypatch.setattr(jobs, "current_depth", lambda: jobs.MAX_DEPTH)
    with pytest.raises(SubagentError) as exc:
        runtime.start(_request())
    assert exc.value.code == "SUBAGENT_DEPTH_EXCEEDED"


def test_a_child_cannot_reach_the_user_or_delegate_again() -> None:
    """The default filter is not arbitrary: each entry is one of those two."""
    for name in ("subagent", "run_background", "ask_user", "notify_user", "watch_create"):
        assert name in DEFAULT_CHILD_DENY


def test_delegated_work_defaults_to_the_background_route() -> None:
    """Which is what puts it on the GPU model when one is configured."""
    assert SubagentRequest(prompt="x").route == "background"


def test_the_in_process_provider_advertises_what_it_supports() -> None:
    from seams.subagent_inproc import InProcessSubagentProvider

    caps = InProcessSubagentProvider().capabilities
    assert caps.tool_filter and caps.route_choice and caps.persona
