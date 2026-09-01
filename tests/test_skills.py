"""Skill discovery, invocation policy, loading, and `/name` invocation."""

from __future__ import annotations

import pytest

from seams.skills import (
    RANK_BUNDLED,
    RANK_PROJECT,
    RANK_USER,
    LocalSkillProvider,
    SkillRegistry,
    SkillRoot,
    SkillSummary,
    find_invocations,
    parse_policy,
    render_skill,
)


def _write(root, name: str, frontmatter: str, body: str = "Do the thing.", bundle: bool = True):
    if bundle:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "SKILL.md"
    else:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.md"
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return path


@pytest.fixture()
def registry(tmp_path):
    def _build(*roots: SkillRoot) -> SkillRegistry:
        reg = SkillRegistry()
        reg.register_provider(LocalSkillProvider(list(roots)))
        return reg

    return _build


@pytest.fixture()
def user_root(tmp_path):
    return SkillRoot(tmp_path / "user", "user", RANK_USER)


# ── discovery ────────────────────────────────────────────────────────────


def test_a_bundle_and_a_flat_file_are_both_skills(registry, user_root) -> None:
    _write(user_root.path, "deploy-api", "description: Ship the API")
    _write(user_root.path, "write-notes", "description: Take notes", bundle=False)
    assert [s.name for s in registry(user_root).list()] == ["deploy-api", "write-notes"]


def test_the_catalog_carries_descriptions_not_bodies(registry, user_root) -> None:
    _write(user_root.path, "deploy-api", "description: Ship the API", body="SECRET STEPS")
    summary = registry(user_root).list()[0]
    assert summary.description == "Ship the API"
    assert type(summary) is SkillSummary
    assert "SECRET STEPS" not in repr(summary)


def test_a_non_kebab_name_is_skipped(registry, user_root) -> None:
    _write(user_root.path, "Deploy_API", "description: Ship it")
    assert registry(user_root).list() == []


def test_a_project_skill_outranks_a_user_skill(tmp_path, registry) -> None:
    project = SkillRoot(tmp_path / "project", "project", RANK_PROJECT)
    user = SkillRoot(tmp_path / "user", "user", RANK_USER)
    _write(project.path, "deploy-api", "description: Project version")
    _write(user.path, "deploy-api", "description: User version")

    winner = registry(project, user).list()[0]
    assert winner.description == "Project version"
    assert winner.source == "project"


def test_a_missing_root_is_not_an_error(tmp_path, registry) -> None:
    assert registry(SkillRoot(tmp_path / "nope", "user", RANK_USER)).list() == []


def test_a_long_description_is_capped(registry, user_root) -> None:
    _write(user_root.path, "verbose", f"description: {'word ' * 300}")
    assert len(registry(user_root).list()[0].description) <= 500


# ── invocation policy ────────────────────────────────────────────────────


def test_omitted_controls_permit_both() -> None:
    policy = parse_policy({"description": "x"})
    assert policy.model_invocable and policy.user_invocable


def test_enabled_false_turns_a_skill_off_entirely() -> None:
    """Keylane documented this field for a long time without reading it."""
    policy = parse_policy({"enabled": "false"})
    assert not policy.model_invocable and not policy.user_invocable


def test_a_model_disabled_skill_stays_user_invocable() -> None:
    policy = parse_policy({"disable-model-invocation": "true"})
    assert not policy.model_invocable
    assert policy.user_invocable


def test_a_disabled_skill_is_absent_from_the_model_catalog(registry, user_root) -> None:
    _write(user_root.path, "off", "description: Off\nenabled: false")
    _write(user_root.path, "on", "description: On")
    reg = registry(user_root)
    assert [s.name for s in reg.for_model()] == ["on"]
    assert [s.name for s in reg.list()] == ["off", "on"]


def test_a_model_disabled_skill_is_hidden_from_the_model_but_not_the_user(
    registry, user_root
) -> None:
    _write(user_root.path, "manual", "description: Manual\ndisable-model-invocation: true")
    reg = registry(user_root)
    assert reg.for_model() == []
    assert [s.name for s in reg.for_user()] == ["manual"]


# ── loading ──────────────────────────────────────────────────────────────


def test_a_loaded_skill_arrives_in_an_envelope(registry, user_root) -> None:
    _write(user_root.path, "deploy-api", "description: Ship it", body="Run make release.")
    rendered = render_skill(registry(user_root).get("deploy-api"))
    assert '<skill_content name="deploy-api">' in rendered
    assert "<skill_instructions>" in rendered
    assert "Run make release." in rendered


def test_the_envelope_names_the_base_directory_for_relative_paths(registry, user_root) -> None:
    """Without it a skill cannot reference the script sitting beside it."""
    _write(user_root.path, "deploy-api", "description: Ship it")
    rendered = render_skill(registry(user_root).get("deploy-api"))
    assert "Base directory for this skill:" in rendered
    assert str(user_root.path / "deploy-api") in rendered


def test_loading_an_unknown_skill_returns_nothing(registry, user_root) -> None:
    assert registry(user_root).get("nonesuch") is None


def test_the_tool_refuses_a_model_disabled_skill(registry, user_root, monkeypatch) -> None:
    _write(user_root.path, "manual", "description: Manual\ndisable-model-invocation: true")
    reg = registry(user_root)
    monkeypatch.setattr("tools.builtin._skill_registry", lambda: reg)
    from tools.builtin import _skill_read

    assert "not available for you to load" in _skill_read("manual")


def test_the_tool_rejects_a_malformed_name(registry, user_root, monkeypatch) -> None:
    monkeypatch.setattr("tools.builtin._skill_registry", lambda: registry(user_root))
    from tools.builtin import _skill_read

    assert 'invalid skill name "../../etc/passwd"' in _skill_read("../../etc/passwd")


# ── user invocation ──────────────────────────────────────────────────────


def test_a_slash_name_invokes_a_user_invocable_skill(registry, user_root) -> None:
    _write(user_root.path, "deploy-api", "description: Ship it")
    skills = registry(user_root).list()
    assert find_invocations("run /deploy-api now please", skills) == ["deploy-api"]


def test_a_slash_name_reaches_a_model_disabled_skill(registry, user_root) -> None:
    """This is the only path to one, which is the point of the separate control."""
    _write(user_root.path, "manual", "description: Manual\ndisable-model-invocation: true")
    assert find_invocations("/manual", registry(user_root).list()) == ["manual"]


def test_an_unknown_slash_word_stays_ordinary_prose(registry, user_root) -> None:
    _write(user_root.path, "deploy-api", "description: Ship it")
    skills = registry(user_root).list()
    assert find_invocations("look in /usr/local and /etc", skills) == []


def test_a_fully_disabled_skill_cannot_be_invoked(registry, user_root) -> None:
    _write(user_root.path, "off", "description: Off\nenabled: false")
    assert find_invocations("/off", registry(user_root).list()) == []


def test_one_skill_is_injected_once_however_often_it_is_named(registry, user_root) -> None:
    _write(user_root.path, "deploy-api", "description: Ship it")
    skills = registry(user_root).list()
    assert find_invocations("/deploy-api then /deploy-api again", skills) == ["deploy-api"]


def test_a_slash_inside_a_path_is_not_an_invocation(registry, user_root) -> None:
    _write(user_root.path, "deploy-api", "description: Ship it")
    skills = registry(user_root).list()
    assert find_invocations("see docs/deploy-api for details", skills) == []


# ── the shipped example ──────────────────────────────────────────────────


def test_the_shipped_example_skill_is_discovered_and_disabled() -> None:
    """It was a flat file under a bundle-only glob, so it was unreachable."""
    from seams import build_context

    skills = {s.name: s for s in build_context().skills.list()}
    assert "example-projects" in skills
    assert not skills["example-projects"].invocation.model_invocable
