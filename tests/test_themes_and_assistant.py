"""Tests for the popup theme spec, the assistant loop, and settings persistence."""

from __future__ import annotations

import json

import pytest

from app.assistant import _extract_json, _heuristic_plan
from app.settings_store import set_toml_key
from app.skills import Skill, SkillRegistry, load_skill_file
from app.themes import POPUP_PRESETS, PopupSpec, parse_popup

# ---------------------------------------------------------------- popup spec


def test_presets_produce_their_advertised_shape():
    assert parse_popup({"preset": "spotlight"}).mode == "bar"
    assert parse_popup({"preset": "panel"}).mode == "panel"
    assert parse_popup({"preset": "window"}).mode == "window"
    assert parse_popup({"preset": "orb"}).mode == "orb"


def test_spotlight_preset_is_chromeless_and_floats_above_centre():
    spec = parse_popup({"preset": "spotlight"})
    assert spec.decorated is False
    assert spec.position == "center"
    assert spec.offset_y < 0
    assert spec.show_status_chips is False


def test_window_preset_is_decorated_and_sticky():
    spec = parse_popup({"preset": "window"})
    assert spec.decorated is True
    assert spec.dismiss_on_focus_loss is False
    assert spec.show_title is True


def test_theme_overrides_win_over_the_preset():
    spec = parse_popup({"preset": "spotlight", "width": 999, "font_size": 30})
    assert spec.mode == "bar"          # from the preset
    assert spec.width == 999           # overridden
    assert spec.font_size == 30


def test_unknown_mode_falls_back_rather_than_raising():
    assert parse_popup({"mode": "hologram"}).mode == "panel"


def test_unknown_position_falls_back():
    assert parse_popup({"position": "sideways"}).position == "center"


def test_opacity_is_clamped():
    assert parse_popup({"opacity": 5}).opacity == 1.0
    assert parse_popup({"opacity": 0.0}).opacity == 0.3
    assert parse_popup({"opacity": "nonsense"}).opacity == 1.0


def test_unknown_keys_are_ignored_not_fatal():
    spec = parse_popup({"preset": "panel", "wobble": True, "width": 640})
    assert spec.width == 640


def test_unknown_preset_falls_back_to_defaults():
    assert parse_popup({"preset": "nope"}).mode == PopupSpec().mode


def test_every_preset_validates():
    for name, values in POPUP_PRESETS.items():
        spec = parse_popup({"preset": name})
        assert spec.width > 0, name
        assert spec.mode in {"bar", "panel", "window", "orb"}, name


# ------------------------------------------------------------ assistant JSON


def test_extract_json_handles_a_plain_object():
    data = _extract_json('{"action": "final", "answer": "done"}')
    assert data["action"] == "final"


def test_extract_json_strips_code_fences():
    data = _extract_json('```json\n{"action": "final", "answer": "hi"}\n```')
    assert data["answer"] == "hi"


def test_extract_json_ignores_prose_after_the_object():
    data = _extract_json('{"action": "ask", "question": "which one?"} Hope that helps!')
    assert data["question"] == "which one?"


def test_extract_json_handles_nested_objects_and_braces_in_strings():
    raw = '{"action":"tool","tool":"write_file","arguments":{"content":"a } b {"}}'
    data = _extract_json(raw)
    assert data["arguments"]["content"] == "a } b {"


def test_extract_json_rejects_output_with_no_object():
    with pytest.raises(ValueError):
        _extract_json("I am afraid I cannot do that.")


# ------------------------------------------------------- heuristic fallback


@pytest.mark.parametrize(
    "message,tool",
    [
        ("open firefox", "open_application"),
        ("launch the calculator", "open_application"),
        ("search the web for openvino", "web_search"),
        ("google fedora 44", "web_search"),
        ("open https://example.com", "open_url"),
        ("system status", "system_info"),
    ],
)
def test_heuristic_recognises_obvious_desktop_intents(message, tool):
    plan = _heuristic_plan(message, message, None)
    assert plan is not None, message
    assert plan[0] == tool


@pytest.mark.parametrize(
    "message",
    [
        "refactor the authentication module",
        "write me a poem about rain",
        "generate an image of a city",
        "why is my build failing",
    ],
)
def test_heuristic_defers_anything_it_should_not_guess(message):
    # Returning None means "fall through to the normal worker router".
    assert _heuristic_plan(message, message, None) is None


# ---------------------------------------------------------------- TOML write


SAMPLE = """[gateway]
host = "127.0.0.1"
port = 9100
local_only = false

[security]
allowed_project_roots = [
  "~/code",
]
require_confirmation_for_modifications = true

[audio]
sample_rate = 16000
"""


def test_set_toml_key_updates_in_place():
    out = set_toml_key(SAMPLE, "gateway", "port", "9200")
    import tomllib

    assert tomllib.loads(out)["gateway"]["port"] == 9200


def test_set_toml_key_appends_into_the_right_section():
    import tomllib

    out = set_toml_key(SAMPLE, "gateway", "docs_url", '"https://docs.example.com"')
    data = tomllib.loads(out)
    assert data["gateway"]["docs_url"] == "https://docs.example.com"
    # The regression this guards: a new key landing in the last table instead.
    assert "docs_url" not in data["audio"]


def test_set_toml_key_creates_a_missing_section():
    import tomllib

    out = set_toml_key(SAMPLE, "assistant", "enabled", "true")
    assert tomllib.loads(out)["assistant"]["enabled"] is True


def test_set_toml_key_replaces_a_multiline_array():
    import tomllib

    out = set_toml_key(
        SAMPLE, "security", "allowed_project_roots", '[\n  "~/a",\n  "~/b",\n]'
    )
    data = tomllib.loads(out)
    assert data["security"]["allowed_project_roots"] == ["~/a", "~/b"]
    assert data["security"]["require_confirmation_for_modifications"] is True


# -------------------------------------------------------------------- skills


def test_skill_front_matter_is_parsed(tmp_path):
    path = tmp_path / "deploy.md"
    path.write_text(
        "---\n"
        "name: deploy\n"
        "description: How to ship.\n"
        "triggers: deploy, ship, release\n"
        "---\n\n"
        "Always use make release.\n",
        encoding="utf-8",
    )
    skill = load_skill_file(path)
    assert skill is not None
    assert skill.name == "deploy"
    assert skill.triggers == ["deploy", "ship", "release"]
    assert "make release" in skill.content


def test_skill_matches_on_a_trigger_word():
    skill = Skill(name="deploy", triggers=["deploy", "ship"], content="x")
    assert skill.matches("please deploy the API") is True
    assert skill.matches("what is the weather") is False


def test_always_on_skill_matches_anything():
    skill = Skill(name="rules", always=True, content="x")
    assert skill.matches("literally anything") is True


def test_disabled_skill_never_matches():
    skill = Skill(name="rules", always=True, enabled=False, content="x")
    assert skill.matches("anything") is False


def test_skill_without_triggers_or_always_never_fires():
    skill = Skill(name="orphan", content="x")
    assert skill.matches("anything") is False


def test_registry_builds_a_prompt_section(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\nname: images\ntriggers: image\n---\nUse flux.\n", encoding="utf-8"
    )
    registry = SkillRegistry(tmp_path)
    section = registry.prompt_section("generate an image of a cat")
    assert "Use flux." in section
    assert registry.prompt_section("what time is it") == ""


# ------------------------------------------------ model output interpretation
# Regression tests for shapes a small model actually emits. Being strict about
# one canonical envelope silently discarded valid tool calls.

from app.assistant import coerce_arguments, interpret_decision  # noqa: E402

KNOWN = {"run_command", "system_info", "web_search", "open_application"}


@pytest.mark.parametrize(
    "decision,expected_tool,expected_args",
    [
        # The documented shape.
        ({"action": "tool", "tool": "system_info", "arguments": {}}, "system_info", {}),
        # Synonyms for the action label.
        ({"action": "call_tool", "tool": "system_info"}, "system_info", {}),
        ({"action": "use_tool", "tool": "web_search", "arguments": {"query": "x"}},
         "web_search", {"query": "x"}),
        # The tool name used as the action, arguments inlined — what Qwen2.5
        # actually produced, and what used to be thrown away.
        ({"action": "run_command", "command": "df -h", "cwd": "/"},
         "run_command", {"command": "df -h", "cwd": "/"}),
        # No action at all, arguments inlined.
        ({"tool": "web_search", "query": "openvino"}, "web_search", {"query": "openvino"}),
        # Arguments as a JSON string.
        ({"action": "tool", "tool": "web_search", "arguments": '{"query": "y"}'},
         "web_search", {"query": "y"}),
        # "parameters" instead of "arguments".
        ({"action": "tool", "tool": "web_search", "parameters": {"query": "z"}},
         "web_search", {"query": "z"}),
    ],
)
def test_tool_calls_are_recognised_in_every_shape(decision, expected_tool, expected_args):
    action, tool, args = interpret_decision(decision, KNOWN)
    assert action == "tool"
    assert tool == expected_tool
    assert args == expected_args


def test_thought_is_never_mistaken_for_an_argument():
    _, _, args = interpret_decision(
        {"action": "run_command", "thought": "check disk", "command": "df"}, KNOWN
    )
    assert args == {"command": "df"}


@pytest.mark.parametrize(
    "decision,expected",
    [
        ({"action": "final", "answer": "done"}, "final"),
        ({"action": "respond", "answer": "done"}, "final"),
        ({"answer": "done"}, "final"),
        ({"action": "ask", "question": "which?"}, "ask"),
        ({"question": "which?"}, "ask"),
        ({"action": "nonsense"}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_non_tool_actions(decision, expected):
    assert interpret_decision(decision, KNOWN)[0] == expected


def test_an_unknown_action_name_is_not_treated_as_a_tool():
    action, tool, _ = interpret_decision({"action": "teleport"}, KNOWN)
    assert action == "unknown" and tool == ""


def test_coerce_arguments():
    assert coerce_arguments({"a": 1}) == {"a": 1}
    assert coerce_arguments('{"a": 1}') == {"a": 1}
    assert coerce_arguments("not json") == {}
    assert coerce_arguments(None) == {}


# ------------------------------------------------------- model directory state

def test_model_status_requires_weights_not_just_a_graph(tmp_path):
    from app.npu.pipeline import model_status

    assert model_status(tmp_path)[0] is False          # empty
    (tmp_path / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    ready, reason = model_status(tmp_path)
    # The graph alone must not count — that is a half-finished download.
    assert ready is False
    assert "openvino_model.bin" in reason

    (tmp_path / "openvino_model.bin").write_bytes(b"\0" * 8192)
    assert model_status(tmp_path) == (True, "ready")


def test_model_status_names_an_interrupted_download(tmp_path):
    from app.npu.pipeline import model_status

    (tmp_path / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    cache = tmp_path / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "blob.incomplete").write_bytes(b"\0" * 2_000_000)
    ready, reason = model_status(tmp_path)
    assert ready is False
    assert "has not finished" in reason and "resumes" in reason


def test_model_status_ignores_tokenizer_graphs(tmp_path):
    from app.npu.pipeline import model_status

    # A tokenizer pair alone is not a model.
    (tmp_path / "openvino_tokenizer.xml").write_text("<net/>", encoding="utf-8")
    (tmp_path / "openvino_tokenizer.bin").write_bytes(b"\0" * 8192)
    assert model_status(tmp_path)[0] is False


# --------------------------------------------------------- prompt size budget

def test_tool_catalogue_stays_small_enough_for_a_1_5b_model():
    from app.tools.registry import ToolRegistry

    registry = ToolRegistry()
    catalogue = registry.prompt_catalog()
    # The budget bounds the plugin tail; built-ins are always listed. An
    # 8,000-token catalogue stopped the model following instructions at all.
    assert len(catalogue) < 3500, f"catalogue grew to {len(catalogue)} chars"
    # Delegation must never be crowded out by a chatty MCP server.
    assert "delegate_to_worker" in catalogue
    assert "list_workers" in catalogue


def test_a_chatty_plugin_cannot_crowd_out_the_built_ins():
    from app.tools.base import BaseTool, ToolDanger, ToolResult
    from app.tools.registry import ToolRegistry

    class Noisy(BaseTool):
        danger = ToolDanger.SAFE
        category = "mcp:noisy"

        def __init__(self, index: int) -> None:
            self.name = f"noisy.tool_{index}"
            self.description = "A very chatty description. " + ("filler " * 60)

        async def run(self, args):  # pragma: no cover - never invoked
            return ToolResult.success("")

    registry = ToolRegistry()
    registry._plugin_tools = {f"noisy.tool_{i}": Noisy(i) for i in range(40)}
    catalogue = registry.prompt_catalog()

    assert "delegate_to_worker" in catalogue
    assert "system_info" in catalogue
    assert len(catalogue) < 3500
    # And it should say plainly that some were left out.
    assert "more plugin tools exist" in catalogue


def test_prompt_lines_are_trimmed_to_one_sentence():
    from app.tools.base import ToolSpec

    spec = ToolSpec(
        name="x",
        description="First sentence here. " + ("padding " * 80),
        parameters={"type": "object", "properties": {}},
    )
    line = spec.prompt_line()
    assert len(line) < 200
    assert "First sentence here." in line


# ------------------------------------------------------------------- canvas

from app.canvas import canvas_from_text, parse_canvas, render_html  # noqa: E402


def test_canvas_parses_the_documented_shape():
    canvas = parse_canvas(
        {
            "title": "Disk usage",
            "summary": "254 GB free.",
            "blocks": [
                {"type": "stats", "items": [{"label": "Free", "value": "254 GB"}]},
                {"type": "table", "columns": ["Mount"], "rows": [["/"]]},
            ],
        }
    )
    assert canvas is not None
    assert canvas.title == "Disk usage"
    assert [b.type for b in canvas.blocks] == ["stats", "table"]


def test_canvas_drops_empty_blocks():
    canvas = parse_canvas(
        {"title": "X", "blocks": [
            {"type": "text", "text": ""},
            {"type": "table", "rows": []},
            {"type": "text", "text": "real"},
        ]}
    )
    # A canvas exists to show content; placeholders are worse than nothing.
    assert [b.type for b in canvas.blocks] == ["text"]


def test_canvas_accepts_a_fenced_block():
    canvas = parse_canvas('Sure:\n```json\n{"title":"T","blocks":[{"type":"text","text":"b"}]}\n```')
    assert canvas is not None and canvas.title == "T"


@pytest.mark.parametrize("payload", ["just prose", "{not json", "", None, 42, {"foo": 1}])
def test_canvas_refuses_anything_that_is_not_one(payload):
    # Must fall back to plain text rather than raise.
    assert parse_canvas(payload) is None


def test_canvas_coerces_loose_types():
    canvas = parse_canvas(
        {"blocks": [{"type": "table", "columns": [1, 2], "rows": [[1, None]]}]}
    )
    assert canvas.blocks[0].columns == ["1", "2"]
    assert canvas.blocks[0].rows == [["1", "None"]]


def test_unknown_note_style_falls_back():
    canvas = parse_canvas({"blocks": [{"type": "note", "text": "x", "style": "chartreuse"}]})
    assert canvas.blocks[0].style == "info"


def test_columnar_output_becomes_a_code_block():
    canvas = canvas_from_text(
        "Filesystem  Size  Used\n/dev/dm-0   952G  695G\ntmpfs        32G  9.2M"
    )
    assert [b.type for b in canvas.blocks] == ["code"]


def test_prose_becomes_paragraphs():
    canvas = canvas_from_text("I opened Firefox.\n\nIt is on screen now.")
    assert [b.type for b in canvas.blocks] == ["text", "text"]


def test_canvas_renders_html_and_escapes_it():
    canvas = parse_canvas(
        {"title": "<script>", "blocks": [{"type": "text", "text": "a & b"}]}
    )
    html = render_html(canvas)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html and "a &amp; b" in html


def test_canvas_round_trips_to_text():
    canvas = parse_canvas(
        {"title": "T", "blocks": [
            {"type": "stats", "items": [{"label": "Free", "value": "2 GB"}]},
            {"type": "list", "entries": ["one", "two"]},
        ]}
    )
    text = canvas.to_text()
    assert "Free: 2 GB" in text and "one" in text


# ------------------------------------------------------------ skill import

from app.skill_import import _looks_like_skill, parse_repo  # noqa: E402


@pytest.mark.parametrize(
    "reference,expected",
    [
        ("owner/repo", ("owner", "repo", None)),
        ("https://github.com/owner/repo", ("owner", "repo", None)),
        ("https://github.com/owner/repo.git", ("owner", "repo", None)),
        ("https://github.com/owner/repo/tree/dev", ("owner", "repo", "dev")),
        ("git@github.com:owner/repo.git", ("owner", "repo", None)),
        ("owner/repo@v2", ("owner", "repo", "v2")),
    ],
)
def test_repo_references_are_parsed(reference, expected):
    assert parse_repo(reference) == expected


def test_bad_repo_reference_is_rejected():
    from app.skill_import import SkillImportError

    for bad in ("", "   ", "justaname"):
        with pytest.raises(SkillImportError):
            parse_repo(bad)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("skills/deploy/SKILL.md", True),
        ("skills/deploy.md", True),
        (".cursor/rules/style.md", True),
        ("agents/planner.md", True),
        # Repo documentation is not a skill.
        ("README.md", False),
        ("skills/README.md", False),
        ("docs/guide.md", False),
        # Never walk into dependency trees.
        ("node_modules/pkg/SKILL.md", False),
        ("src/app.py", False),
    ],
)
def test_skill_file_detection(path, expected):
    assert _looks_like_skill(path) is expected


def test_front_matter_block_scalars_are_folded():
    from app.skill_import import _metadata

    text = "---\nname: demo\ndescription: >\n  A folded\n  description.\n---\n\nBody."
    name, description, _ = _metadata(text, "skills/demo/SKILL.md")
    assert name == "demo"
    assert description == "A folded description."


def test_skill_name_falls_back_to_the_folder():
    from app.skill_import import _metadata

    name, _, _ = _metadata("no front matter", "skills/deploy/SKILL.md")
    assert name == "deploy"


# --------------------------------------------------------------------- tts

from app.tts import clean_for_speech, probe_engines, resolve  # noqa: E402


def test_speech_text_is_stripped_of_markup():
    cleaned = clean_for_speech("**Disk** is `74%` full.\n\n- root\n- boot")
    # A synthesiser should read words, not asterisks and backticks.
    assert "*" not in cleaned and "`" not in cleaned
    assert "Disk" in cleaned and "74%" in cleaned


def test_speech_text_replaces_urls_and_code_blocks():
    assert "a link" in clean_for_speech("See https://example.com/x for more")
    assert "code block" in clean_for_speech("Run:\n```\nls -la\n```\nDone")


def test_speech_text_is_capped_at_a_sentence():
    body = ("This is a sentence. " * 200).strip()
    cut = clean_for_speech(body, limit=200)
    assert len(cut) <= 200
    assert cut.endswith(".")


def test_engine_probe_reports_every_engine():
    engines = {e.id for e in probe_engines(refresh=True)}
    # Present or not, all three must be described so the panel can explain.
    assert engines == {"piper", "espeak", "flite"}


def test_unavailable_engine_names_how_to_install_it():
    for engine in probe_engines():
        if not engine.available:
            assert engine.install_hint, f"{engine.id} gives no way to fix it"


def test_resolve_falls_back_when_the_engine_is_missing():
    engine, _voice = resolve("does-not-exist", "")
    available = [e for e in probe_engines() if e.available]
    if available:
        assert engine is not None and engine.available
    else:
        assert engine is None


# --------------------------------------------------- result orb behaviour

from launcher_stubs import canvas_text, panel_width  # noqa: E402


def test_spoken_canvas_describes_tables_rather_than_reading_them():
    text = canvas_text(
        {"title": "Disk", "blocks": [
            {"type": "table", "columns": ["a"], "rows": [["1"], ["2"]]},
            {"type": "code", "text": "df -h"},
        ]}
    )
    assert "table of 2 rows" in text
    assert "df -h" not in text            # reading raw output aloud is useless
    assert ".." not in text               # doubled stops make it stumble


def test_spoken_canvas_includes_stats_and_prose():
    text = canvas_text(
        {"summary": "All good.", "blocks": [
            {"type": "stats", "items": [{"label": "Free", "value": "2 GB"}]},
            {"type": "text", "text": "Nothing to do."},
        ]}
    )
    assert "Free: 2 GB" in text and "Nothing to do" in text


def test_short_answers_get_a_narrow_panel():
    short = {"blocks": [{"type": "text", "text": "It is 9:40 PM."}]}
    wide = {"blocks": [{"type": "table", "columns": ["a"], "rows": [["1"]]}]}
    # A one-liner in a full-width panel is mostly empty panel.
    assert panel_width(short, canvas_text(short)) < panel_width(wide, canvas_text(wide))


# ------------------------------------------------ structural canvas building

from app.canvas_build import markdown_to_canvas, output_to_canvas  # noqa: E402


@pytest.mark.parametrize(
    "label,text,columns",
    [
        (
            "df -h",
            "Filesystem      Size  Used Avail Use% Mounted on\n"
            "/dev/dm-0       952G  732G  217G  78% /\n"
            "tmpfs            32G   25M   32G   1% /dev/shm",
            ["Filesystem", "Size", "Used", "Avail", "Use%", "Mounted on"],
        ),
        (
            "ps",
            "  PID TTY          TIME CMD\n"
            " 1234 pts/0    00:00:01 bash\n"
            " 5678 pts/0    00:00:00 python",
            ["PID", "TTY", "TIME", "CMD"],
        ),
    ],
)
def test_columnar_output_becomes_a_table(label, text, columns):
    # This is what makes the popup useful when the model never emits a canvas.
    block = output_to_canvas(text, command=label).blocks[0]
    assert block.type == "table", f"{label} did not become a table"
    assert block.columns == columns
    assert all(len(row) == len(columns) for row in block.rows)


@pytest.mark.parametrize(
    "text",
    [
        "I opened Firefox for you.",
        "ERROR failed\nWARN thing\nINFO ok",
    ],
)
def test_non_tabular_output_is_not_forced_into_a_table(text):
    assert output_to_canvas(text).blocks[0].type in {"text", "code"}


def test_markdown_becomes_blocks_not_literal_characters():
    canvas = markdown_to_canvas(
        "## Disk usage\n\n"
        "You have 217 GB free.\n\n"
        "| Mount | Use% |\n|-------|------|\n| /     | 78%  |\n\n"
        "> **Warning** Root is nearly full.\n\n"
        "```shell\ndf -h\n```"
    )
    kinds = [b.type for b in canvas.blocks]
    assert canvas.title == "Disk usage"          # the H2 became the title
    assert canvas.summary.startswith("You have")  # short lead became the summary
    assert "table" in kinds and "note" in kinds and "code" in kinds
    note = next(b for b in canvas.blocks if b.type == "note")
    assert note.style == "warning"
    # None of the markup should survive as literal text.
    rendered = canvas.to_text()
    assert "##" not in rendered and "|---" not in rendered and "**" not in rendered


def test_a_list_of_pairs_becomes_stats():
    canvas = markdown_to_canvas("- Free: 217 GB\n- Used: 732 GB")
    assert canvas.blocks[0].type == "stats"
    assert canvas.blocks[0].items[0].label == "Free"


def test_a_plain_list_stays_a_list():
    canvas = markdown_to_canvas("- open the file\n- read it\n- close it")
    assert canvas.blocks[0].type == "list"


def test_inline_markup_is_stripped():
    canvas = markdown_to_canvas("Some **bold** and `code` and *italic* text here.")
    body = canvas.summary or canvas.blocks[0].text
    assert "*" not in body and "`" not in body
    assert "bold" in body and "code" in body


# ----------------------------------------------------------- loader states

from launcher_stubs import loader_states  # noqa: E402


def test_every_assistant_state_has_a_colour():
    states = loader_states()
    for needed in ("thinking", "tool", "delegating", "verifying", "waiting", "failed", "done"):
        assert needed in states, f"no colour for {needed}"


def test_state_colours_are_distinguishable():
    states = loader_states()
    # Colours that differ by less than this read as the same hue in a 26px orb.
    values = list(states.values())
    for i, a in enumerate(values):
        for b in values[i + 1 :]:
            distance = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
            assert distance > 0.12, f"{a} and {b} are too close to tell apart"


# ------------------------------------------------------ gguf quant choice

from app.hardware import HardwareProfile  # noqa: E402
from app.hf_hub import _gguf_allow_patterns  # noqa: E402

_REPO = [
    "Model-Q2_K.gguf",
    "Model-Q4_K_M.gguf",
    "Model-Q5_K_M.gguf",
    "Model-Q8_0.gguf",
    "mmproj-F16.gguf",
    "README.md",
]


def _hw(vram: int) -> HardwareProfile:
    profile = HardwareProfile()
    profile.nvidia_vram_mb = vram
    return profile


@pytest.mark.parametrize(
    "vram,expected",
    [(24000, "Model-Q5_K_M.gguf"), (12000, "Model-Q4_K_M.gguf"), (4000, "Model-Q2_K.gguf")],
)
def test_one_quant_is_chosen_for_the_hardware(vram, expected):
    patterns = _gguf_allow_patterns(_hw(vram), None, _REPO)
    weights = [p for p in patterns if p.endswith(".gguf") and "mmproj" not in p]
    # Passing several globs downloads several copies of the same model.
    assert weights == [expected]


def test_the_projector_travels_with_its_model():
    patterns = _gguf_allow_patterns(_hw(24000), None, _REPO)
    assert "mmproj-F16.gguf" in patterns


def test_an_explicit_filename_wins():
    patterns = _gguf_allow_patterns(_hw(24000), "Model-Q8_0.gguf", _REPO)
    assert patterns[0] == "Model-Q8_0.gguf"


def test_without_a_listing_it_still_picks_one_pattern():
    patterns = _gguf_allow_patterns(_hw(24000), None, None)
    globs = [p for p in patterns if p.endswith(".gguf")]
    assert len(globs) == 1
