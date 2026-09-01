"""Model routing, tool scoping, and the execution pipeline."""

from __future__ import annotations

import asyncio
import json

import pytest

from seams.errors import LlmError
from seams.llm import LlmRuntime
from tools.registry import Tool, ToolOutcome, ToolRegistry, deny


# ── model routing ────────────────────────────────────────────────────────


class FakeAdapter:
    def __init__(self, adapter_id: str, ready: bool = True) -> None:
        self.id = adapter_id
        self.ready = ready
        self.calls: list[str] = []

    def available(self) -> bool:
        return self.ready

    @property
    def status(self):
        return {"kind": "fake"}

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return f"{self.id}:{prompt}"

    def chat(self, messages, **kwargs) -> str:
        return f"{self.id}:chat"


@pytest.fixture()
def llm():
    runtime = LlmRuntime()
    runtime.configure_routes({"interactive": ["npu"], "background": ["gpu", "npu"]})
    return runtime


def test_a_route_prefers_its_first_available_adapter(llm) -> None:
    npu, gpu = FakeAdapter("npu"), FakeAdapter("gpu")
    llm.register(npu)
    llm.register(gpu)
    assert llm.generate("x", route="background") == "gpu:x"
    assert llm.generate("x", route="interactive") == "npu:x"


def test_a_route_falls_through_an_unavailable_adapter(llm) -> None:
    """A GPU model that is configured off must not strand background work."""
    llm.register(FakeAdapter("npu"))
    llm.register(FakeAdapter("gpu", ready=False))
    assert llm.generate("x", route="background") == "npu:x"


def test_resolution_does_not_depend_on_registration_order(llm) -> None:
    """The preference list decides, not which plugin loaded first."""
    llm.register(FakeAdapter("gpu"))
    llm.register(FakeAdapter("npu"))
    assert llm.resolve("background").id == "gpu"

    other = LlmRuntime()
    other.configure_routes({"background": ["gpu", "npu"]})
    other.register(FakeAdapter("npu"))
    other.register(FakeAdapter("gpu"))
    assert other.resolve("background").id == "gpu"


def test_an_unavailable_route_reports_which_adapters_it_wanted(llm) -> None:
    llm.register(FakeAdapter("npu", ready=False))
    llm.register(FakeAdapter("gpu", ready=False))
    with pytest.raises(LlmError) as exc:
        llm.resolve("background")
    assert exc.value.code == "LLM_ROUTE_UNAVAILABLE"
    assert "gpu" in exc.value.message and "npu" in exc.value.message


def test_a_route_naming_a_missing_adapter_fails_loudly(llm) -> None:
    llm.configure_routes({"background": ["nonesuch"]})
    llm.register(FakeAdapter("npu"))
    with pytest.raises(LlmError) as exc:
        llm.resolve("background")
    assert exc.value.code == "LLM_ROUTE_MISSING"


def test_duplicate_adapter_ids_are_a_programming_error(llm) -> None:
    llm.register(FakeAdapter("npu"))
    with pytest.raises(LlmError, match="already registered"):
        llm.register(FakeAdapter("npu"))


def test_disposing_an_adapter_unregisters_it(llm) -> None:
    dispose = llm.register(FakeAdapter("npu"))
    assert llm.is_ready("interactive")
    dispose()
    assert not llm.is_ready("interactive")


# ── tool scoping ─────────────────────────────────────────────────────────


def _tool(name: str, result: str = "ok") -> Tool:
    return Tool(
        name=name,
        description=name,
        parameters={"type": "object", "properties": {}},
        handler=lambda: result,
    )


@pytest.fixture()
def root():
    reg = ToolRegistry()
    for name in ("recall", "remember", "shell", "web_search"):
        reg.register(_tool(name))
    return reg


def test_a_child_inherits_everything_by_default(root) -> None:
    child = root.child("agent-1")
    assert set(child.visible()) == set(root.visible())


def test_an_allow_list_hides_everything_else(root) -> None:
    child = root.child("agent-1")
    child.restrict(allow=["recall", "web_search"])
    assert set(child.visible()) == {"recall", "web_search"}
    assert root.get("shell") is not None  # the parent is untouched


def test_a_denied_tool_also_refuses_to_run(root) -> None:
    """Visibility and authority are the same thing, as they are in DSH."""
    child = root.child("agent-1")
    child.restrict(deny=["shell"])
    assert child.get("shell") is None
    outcome = asyncio.run(child.execute("shell", {}))
    assert outcome.is_error and outcome.code == "UNKNOWN_TOOL"


def test_a_scopes_own_registration_survives_its_restriction(root) -> None:
    child = root.child("agent-1")
    child.restrict(allow=["recall"])
    child.register(_tool("report_result"))
    assert set(child.visible()) == {"recall", "report_result"}


def test_restrictions_intersect_rather_than_replace(root) -> None:
    child = root.child("agent-1")
    child.restrict(allow=["recall", "remember", "shell"])
    child.restrict(allow=["recall", "shell"])
    child.restrict(deny=["shell"])
    assert set(child.visible()) == {"recall"}


def test_restricting_an_unknown_tool_fails_loudly(root) -> None:
    """A typo that silently filters nothing is a filter that does not exist."""
    with pytest.raises(ValueError, match="nonesuch"):
        root.child("agent-1").restrict(allow=["nonesuch"])


def test_disposing_a_restriction_restores_visibility(root) -> None:
    child = root.child("agent-1")
    dispose = child.restrict(deny=["shell"])
    assert child.get("shell") is None
    dispose()
    assert child.get("shell") is not None


# ── the execution pipeline ───────────────────────────────────────────────


def test_a_pre_hook_can_deny_a_call(root) -> None:
    root.add_pre_hook(lambda call: deny("nope", code="TEST_DENIED") if call.name == "shell" else None)
    outcome = asyncio.run(root.execute("shell", {}))
    assert outcome.is_error and outcome.code == "TEST_DENIED"
    assert json.loads(outcome.content)["error"] == "nope"


def test_a_post_hook_can_replace_the_outcome(root) -> None:
    root.add_post_hook(lambda call, outcome: ToolOutcome(content="replaced"))
    assert asyncio.run(root.execute("recall", {})).content == "replaced"


def test_a_child_inherits_its_parents_hooks(root) -> None:
    seen: list[str] = []
    root.add_pre_hook(lambda call: seen.append(call.name))
    asyncio.run(root.child("agent-1").execute("recall", {}))
    assert seen == ["recall"]


def test_an_unknown_tool_names_what_is_available(root) -> None:
    outcome = asyncio.run(root.execute("nonesuch", {}))
    payload = json.loads(outcome.content)
    assert payload["code"] == "UNKNOWN_TOOL"
    assert "recall" in payload["available"]


def test_a_handler_type_error_reports_the_expected_arguments(root) -> None:
    root.register(
        Tool(
            name="needs_args",
            description="",
            parameters={"type": "object", "properties": {"a": {"type": "string"}}},
            handler=lambda a: a,
        )
    )
    payload = json.loads(asyncio.run(root.execute("needs_args", {"b": 1})).content)
    assert payload["code"] == "INVALID_ARGS"
    assert payload["expected_arguments"] == ["a"]
    assert payload["received"] == ["b"]


def test_a_slow_tool_is_cut_off_at_its_timeout(root) -> None:
    async def _slow() -> str:
        await asyncio.sleep(5)
        return "never"

    root.register(
        Tool(
            name="slow",
            description="",
            parameters={"type": "object", "properties": {}},
            handler=_slow,
            timeout_ms=30,
        )
    )
    outcome = asyncio.run(root.execute("slow", {}))
    assert outcome.is_error and outcome.code == "TOOL_TIMEOUT"


def test_a_seam_error_keeps_its_code(root) -> None:
    def _boom() -> str:
        raise LlmError("LLM_ROUTE_UNAVAILABLE", "no model is loaded")

    root.register(
        Tool(name="boom", description="", parameters={"type": "object", "properties": {}}, handler=_boom)
    )
    outcome = asyncio.run(root.execute("boom", {}))
    assert outcome.code == "LLM_ROUTE_UNAVAILABLE"
    assert json.loads(outcome.content)["error"] == "no model is loaded"


# ── the OpenAI-compatible adapter ────────────────────────────────────────


def _adapter(**kwargs):
    from seams.llm_adapters import OpenAiCompatAdapter

    defaults = dict(
        adapter_id="gpu",
        base_url="http://127.0.0.1:1234/v1",
        model="qwen2.5-14b",
        enabled=True,
    )
    return OpenAiCompatAdapter(**{**defaults, **kwargs})


def test_availability_is_a_local_check_not_a_request() -> None:
    """Route resolution happens on every step; it cannot make a round trip."""
    assert _adapter().available() is True
    assert _adapter(enabled=False).available() is False
    assert _adapter(model="").available() is False
    assert _adapter(base_url="").available() is False


def test_auto_unload_is_off_unless_asked_for() -> None:
    assert _adapter()._idle_fields() == {}


def test_auto_unload_sends_both_known_spellings() -> None:
    """Ollama reads keep_alive, LM Studio reads ttl; neither is standard."""
    fields = _adapter(auto_unload=True, idle_seconds=90)._idle_fields()
    assert fields == {"keep_alive": 90, "ttl": 90}


def test_a_negative_idle_is_clamped() -> None:
    assert _adapter(auto_unload=True, idle_seconds=-5)._idle_fields()["ttl"] == 0


def test_the_request_carries_the_idle_fields(monkeypatch) -> None:
    import httpx

    sent = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "hi"}}]}

    class _Client:
        def __init__(self, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def post(self, url, json=None, headers=None):
            sent.update(json or {})
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    assert _adapter(auto_unload=True, idle_seconds=30).generate("hello") == "hi"
    assert sent["keep_alive"] == 30 and sent["ttl"] == 30

    sent.clear()
    _adapter().generate("hello")
    assert "keep_alive" not in sent and "ttl" not in sent


def test_an_unreachable_server_is_a_structured_error(monkeypatch) -> None:
    import httpx

    class _Client:
        def __init__(self, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def post(self, *_a, **_kw):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "Client", _Client)
    with pytest.raises(LlmError) as exc:
        _adapter().generate("hello")
    assert exc.value.code == "LLM_TRANSPORT_ERROR"
