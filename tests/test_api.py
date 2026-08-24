"""End-to-end API tests against the real app, with no external services."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_control_panel_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Keylane" in response.text


def test_handbook_is_served_at_docs(client):
    response = client.get("/docs/")
    assert response.status_code == 200
    assert "Keylane handbook" in response.text


def test_handbook_pages_cross_link(client):
    for page in ("plugins", "themes", "tools", "assistant", "skills", "api", "install", "popup"):
        response = client.get(f"/docs/{page}.html")
        assert response.status_code == 200, page
        # Every page carries the sidebar, so navigation always works.
        assert 'class="side-links"' in response.text, page


def test_openapi_explorer_moved_aside_for_the_handbook(client):
    assert client.get("/api-docs").status_code == 200


def test_status_reports_the_assistant_surface(client):
    data = client.get("/api/status").json()
    for key in ("npu", "assistant", "tools_enabled", "tool_count", "busy", "local_only"):
        assert key in data, key
    assert isinstance(data["tool_count"], int)


def test_tools_endpoint_lists_built_ins(client):
    data = client.get("/api/tools").json()
    names = {t["name"] for t in data["tools"]}
    for expected in (
        "open_application",
        "web_search",
        "read_file",
        "run_command",
        "delegate_to_worker",
        "list_workers",
        "system_info",
    ):
        assert expected in names, expected


def test_tool_specs_carry_danger_and_schema(client):
    data = client.get("/api/tools").json()
    by_name = {t["name"]: t for t in data["tools"]}
    assert by_name["run_command"]["danger"] == "dangerous"
    assert by_name["system_info"]["danger"] == "safe"
    assert by_name["open_application"]["parameters"]["type"] == "object"


def test_calling_a_safe_tool_works(client):
    response = client.post(
        "/api/tools/call", json={"tool": "system_info", "arguments": {}}
    )
    body = response.json()
    assert body["ok"] is True
    assert "hostname" in body["output"]


def test_calling_a_gated_tool_asks_first(client):
    response = client.post(
        "/api/tools/call",
        json={"tool": "open_application", "arguments": {"application": "nothing-here"}},
    )
    body = response.json()
    assert body["requires_confirmation"] is True
    assert body["tool"] == "open_application"
    assert body["danger"] == "sensitive"


def test_unknown_tool_is_reported_not_crashed(client):
    body = client.post("/api/tools/call", json={"tool": "nope", "arguments": {}}).json()
    assert body["ok"] is False


def test_assistant_endpoint_exposes_the_system_prompt(client):
    data = client.get("/api/assistant").json()
    prompt = data["system_prompt"]
    # The three behaviours the assistant is built around must be in the prompt.
    assert "DO IT YOURSELF" in prompt
    assert "DELEGATE" in prompt
    assert "FOLLOW UP" in prompt
    assert "delegate_to_worker" in prompt
    # And it has to stay small enough for a 1.5B model to actually follow.
    assert len(prompt) < 6000, f"system prompt grew to {len(prompt)} chars"


def test_activity_snapshot_shape(client):
    data = client.get("/api/activity").json()
    for key in ("busy", "active_count", "needs_attention", "active", "recent"):
        assert key in data, key


def test_active_theme_exposes_the_popup_spec(client):
    data = client.get("/api/themes/active").json()
    assert data["popup"]["mode"] in {"bar", "panel", "window", "orb"}
    assert "accent" in data["colors"]


def test_popup_json_endpoint(client):
    spec = client.get("/api/themes/active/popup.json").json()
    assert isinstance(spec["width"], int)
    assert "decorated" in spec


def test_theme_list_includes_every_popup_mode(client):
    themes = client.get("/api/themes").json()
    modes = {t["popup"]["mode"] for t in themes}
    # The whole point of the theme system: all four shapes ship built in.
    assert {"bar", "panel", "window", "orb"} <= modes


def test_default_theme_is_a_spotlight_bar(client):
    themes = {t["id"]: t for t in client.get("/api/themes").json()}
    assert themes["default"]["popup"]["mode"] == "bar"
    assert themes["default"]["popup"]["decorated"] is False


def test_launcher_css_is_generated_for_the_active_theme(client):
    response = client.get("/api/themes/active/launcher.css")
    assert response.status_code == 200
    assert "@define-color ag_accent" in response.text
    assert ".keylane-shell" in response.text


def test_theme_css_for_the_panel(client):
    response = client.get("/theme.css")
    assert response.status_code == 200
    assert "--ag-accent" in response.text


def test_plugins_list_and_kinds(client):
    plugins = client.get("/api/plugins?health=false").json()
    by_id = {p["id"]: p for p in plugins}
    assert "claude" in by_id and by_id["claude"]["cloud"] is True
    assert "lmstudio" in by_id and by_id["lmstudio"]["cloud"] is False
    assert by_id["comfyui"]["kind"] == "mcp"


def test_skills_endpoint(client):
    data = client.get("/api/skills").json()
    assert "skills" in data and "directory" in data


def test_config_round_trips_the_docs_url(client):
    original = client.get("/api/config").json()
    try:
        saved = client.put(
            "/api/config", json={"docs_url": "https://docs.example.test"}
        ).json()
        assert saved["docs_url"] == "https://docs.example.test"
        # And it survives a re-read from disk.
        assert client.get("/api/config").json()["docs_url"] == "https://docs.example.test"
    finally:
        client.put("/api/config", json={"docs_url": original["docs_url"]})


def test_route_endpoint_returns_a_decision(client):
    response = client.post("/api/route", json={"message": "what is 2 + 2"})
    assert response.status_code == 200
    decision = response.json()
    assert decision["worker"]
    assert decision["action"]


# ------------------------------------------------ expanded control surface


def test_system_overview(client):
    d = client.get("/api/system").json()
    for key in ("version", "python", "platform", "paths", "pipelines", "counts"):
        assert key in d, key
    assert {"router", "verifier"} <= set(d["pipelines"])
    assert {"plugins", "tools", "skills", "projects"} <= set(d["counts"])
    for key in ("root", "config", "models", "themes", "skills", "plugins"):
        assert key in d["paths"], key


def test_worker_endpoints_round_trip(client):
    original = client.get("/api/workers").json()["settings"]
    try:
        saved = client.put(
            "/api/workers", json={"lmstudio_timeout_seconds": 199}
        ).json()
        assert "lmstudio.timeout_seconds" in saved["changed"]
        assert saved["settings"]["lmstudio_timeout_seconds"] == 199
        # And it survives a re-read from disk.
        assert client.get("/api/workers").json()["settings"][
            "lmstudio_timeout_seconds"
        ] == 199
    finally:
        client.put(
            "/api/workers",
            json={"lmstudio_timeout_seconds": original["lmstudio_timeout_seconds"]},
        )


def test_worker_endpoint_validation_rejects_nonsense(client):
    assert client.put("/api/workers", json={"audio_channels": 9}).status_code == 422


def test_tool_policy_can_disable_and_restore(client):
    def spec(name):
        tools = client.get("/api/tools").json()["tools"]
        return next(t for t in tools if t["name"] == name)

    assert spec("web_search")["enabled"] is True
    try:
        body = client.post("/api/tools/web_search/policy", json={"enabled": False}).json()
        assert body["tool"]["enabled"] is False
        assert "web_search" in body["policy"]["deny"]
        # The registry must actually refuse it, not just report it.
        called = client.post(
            "/api/tools/call", json={"tool": "web_search", "arguments": {"query": "x"}}
        ).json()
        assert called["ok"] is False
    finally:
        client.post("/api/tools/web_search/policy", json={"enabled": True})
    assert spec("web_search")["enabled"] is True


def test_tool_policy_auto_confirm_removes_the_gate(client):
    try:
        body = client.post(
            "/api/tools/open_application/policy", json={"auto_confirm": True}
        ).json()
        assert body["tool"]["requires_confirmation"] is False
    finally:
        client.post("/api/tools/open_application/policy", json={"auto_confirm": False})


def test_tool_policy_404s_for_unknown_tool(client):
    assert client.post("/api/tools/nope/policy", json={"enabled": False}).status_code == 404


def test_skill_crud_round_trip(client):
    name = "pytest-temp-skill"
    try:
        created = client.post(
            "/api/skills",
            json={
                "name": name,
                "description": "temporary",
                "triggers": ["zzz-trigger"],
                "content": "Body text.",
            },
        ).json()
        assert created["name"] == name
        assert created["triggers"] == ["zzz-trigger"]

        # Duplicate names are refused.
        assert client.post("/api/skills", json={"name": name, "content": "x"}).status_code == 400

        toggled = client.post(f"/api/skills/{name}/enable", json={"enabled": False}).json()
        assert toggled["enabled"] is False

        updated = client.put(
            f"/api/skills/{name}",
            json={"name": name, "content": "New body.", "triggers": ["aaa"], "enabled": True},
        ).json()
        assert updated["content"] == "New body."
        assert updated["enabled"] is True
    finally:
        client.delete(f"/api/skills/{name}")

    names = {s["name"] for s in client.get("/api/skills").json()["skills"]}
    assert name not in names


def test_skill_name_is_validated(client):
    assert client.post("/api/skills", json={"name": "../escape", "content": "x"}).status_code == 400
    assert client.post("/api/skills", json={"name": "", "content": "x"}).status_code == 400


def test_unknown_skill_operations_404(client):
    assert client.delete("/api/skills/not-a-skill").status_code == 404
    assert client.put(
        "/api/skills/not-a-skill", json={"name": "x", "content": "y"}
    ).status_code == 404


def test_projects_reject_paths_outside_the_sandbox(client):
    original = client.get("/api/projects").json()["projects"]
    try:
        result = client.put(
            "/api/projects",
            json={"projects": [{"name": "Etc", "path": "/etc"}]},
        ).json()
        assert result["projects"] == []
        assert result["rejected"] and "outside" in result["rejected"][0]["reason"].lower()
    finally:
        client.put("/api/projects", json={"projects": original})


# ---------------------------------------------- recommended model downloads


def test_every_recommendation_offers_an_action(client):
    """No card may be a dead end.

    A recommendation must be usable: already installed, downloadable, or
    explicitly gated with a link. "Not downloaded" with no way forward is the
    bug this guards.
    """
    import re

    data = client.get("/api/models").json()
    recommendations = data.get("recommendations") or {}
    seen = 0
    for kind in ("router", "chat"):
        for model in recommendations.get(kind) or []:
            seen += 1
            installed = model.get("installed") or model.get("available")
            repo = re.search(r"huggingface\.co/([^/]+/[^/?#]+)", model.get("hf_url") or "")
            assert installed or repo or model.get("gated"), (
                f"{model['id']} offers no action: not installed, no repo, not gated"
            )
    assert seen, "no recommendations to check"


def test_gated_models_are_marked_rather_than_offered(client):
    data = client.get("/api/models").json()
    router = (data.get("recommendations") or {}).get("router") or []
    gated = [m for m in router if m.get("gated")]
    # These repositories 401 without an accepted licence; a download button
    # would fail with no explanation.
    for model in gated:
        assert model.get("hf_url"), f"{model['id']} is gated but has nowhere to send the user"


def test_chat_downloads_target_lm_studio(client):
    from app.hf_hub import lmstudio_models_dir, target_dir

    destination = target_dir("chat", "lmstudio-community/Some-Model-GGUF")
    lmstudio = lmstudio_models_dir()
    if lmstudio is None:
        return  # no LM Studio on this machine; the fallback path is fine
    # LM Studio only scans its own tree, laid out publisher/repo.
    assert str(destination).startswith(str(lmstudio))
    assert destination.name == "Some-Model-GGUF"
    assert destination.parent.name == "lmstudio-community"


def test_models_overview_names_the_real_download_location(client):
    """Relative paths hide where downloads actually go.

    ``./models/router`` reads as the current directory, which for anyone with
    the source checked out is the wrong folder — the service runs from the
    install. The panel has to state the absolute path.
    """
    from pathlib import Path

    data = client.get("/api/models").json()
    paths = data.get("paths") or {}
    for key in ("root", "models", "router", "chat", "resolved_router"):
        assert key in paths, f"no {key} path reported"
        assert Path(paths[key]).is_absolute(), f"{key} is not absolute: {paths[key]}"
    assert paths["router"].endswith("models/router")


def test_installed_models_carry_an_absolute_path(client):
    data = client.get("/api/models").json()
    from pathlib import Path

    for model in (data.get("recommendations") or {}).get("installed") or []:
        assert Path(model["absolute"]).is_absolute()
        # The relative form is still stored, for a portable config.
        assert model["path"].startswith("./")


def test_a_graph_without_weights_is_not_an_installed_model(tmp_path):
    """An interrupted download leaves the .xml and not the .bin.

    The directory looks like a model to anything that only checks for the
    graph, which is how a half-downloaded router ends up reported as
    installed and then fails to load.
    """
    from app.models_catalog import _looks_like_openvino_model, missing_weights

    model = tmp_path / "OpenVINO__Something-int4-ov"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "openvino_model.xml").write_text("<net/>")
    (model / "openvino_detokenizer.xml").write_text("<net/>")
    (model / "openvino_detokenizer.bin").write_bytes(b"\x00" * 8)

    assert missing_weights(model) == ["openvino_model.bin"]
    assert _looks_like_openvino_model(model) is False

    (model / "openvino_model.bin").write_bytes(b"\x00" * 8)
    assert missing_weights(model) == []
    assert _looks_like_openvino_model(model) is True


def test_status_reports_incomplete_models_so_the_panel_can_offer_a_resume(client):
    """The field exists and is a list, whatever this machine has on disk."""
    body = client.get("/api/status").json()
    assert isinstance(body["incomplete_models"], list)
    for entry in body["incomplete_models"]:
        assert entry["missing"], "an incomplete model must name what is missing"
        assert "/" in entry["repo_id"], "a resume needs the repo id"
