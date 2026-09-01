"""Answer canvas: block parsing, source stripping, markup."""

from __future__ import annotations

from ui.canvas import block_markup, headline_text, parse_blocks, plain_text, strip_sources


def kinds(answer: str) -> list[str]:
    return [b.kind for b in parse_blocks(answer)]


# ── sources are never shown ──────────────────────────────────────────────


def test_a_trailing_sources_section_is_removed() -> None:
    text = "Next.js 16.2 is current.\n\nSources\n[1] Next.js Blog — https://nextjs.org\n[2] Other — https://x"
    assert strip_sources(text) == "Next.js 16.2 is current."


def test_a_markdown_sources_heading_is_removed() -> None:
    assert strip_sources("Answer body.\n\n## Sources\n[1] a — https://a") == "Answer body."


def test_a_bold_sources_heading_is_removed() -> None:
    assert strip_sources("Answer body.\n\n**Sources:**\n[1] a — https://a") == "Answer body."


def test_citation_markers_are_stripped() -> None:
    """With no source list on screen, a dangling [2] points at nothing."""
    assert strip_sources("Speed is the headline [1], with 200 fixes [2].") == (
        "Speed is the headline, with 200 fixes."
    )


def test_based_on_n_sources_boilerplate_goes() -> None:
    assert strip_sources("Based on 4 sources:\nThe answer.") == "The answer."


def test_a_body_without_sources_is_untouched() -> None:
    assert strip_sources("Just an answer.") == "Just an answer."


def test_plain_text_is_clipboard_ready() -> None:
    assert "[1]" not in plain_text("Answer [1].\n\nSources\n[1] x — https://x")


# ── block structure ──────────────────────────────────────────────────────


def test_the_opening_paragraph_becomes_the_headline() -> None:
    assert kinds("The answer is 42.\n\nMore prose here.") == ["headline", "text"]


def test_headings_bullets_numbers_code_and_quotes() -> None:
    answer = (
        "Lede sentence.\n\n"
        "## What is new\n"
        "- one\n- two\n\n"
        "### Steps\n"
        "1. first\n2. second\n\n"
        "```bash\nnpm i\n```\n\n"
        "> A caveat.\n"
    )
    assert kinds(answer) == [
        "headline", "heading", "bullets", "heading", "numbers", "code", "quote",
    ]


def test_a_run_of_term_value_bullets_becomes_a_definition_list() -> None:
    blocks = parse_blocks("Lede.\n\n- **Turbopack** — 200 fixes\n- **Startup** — 400% faster")
    kv = [b for b in blocks if b.kind == "kv"]
    assert kv and kv[0].pairs == [("Turbopack", "200 fixes"), ("Startup", "400% faster")]


def test_a_mixed_bullet_list_stays_a_list_in_source_order() -> None:
    blocks = parse_blocks("Lede.\n\n- **Bold** — value\n- plain item\n- another")
    bullets = [b for b in blocks if b.kind == "bullets"][0]
    assert bullets.items == ["**Bold** — value", "plain item", "another"]


def test_code_fences_keep_their_body_and_language() -> None:
    code = [b for b in parse_blocks("Lede.\n\n```bash\nnpm run dev\n```") if b.kind == "code"][0]
    assert code.language == "bash"
    assert code.text == "npm run dev"


def test_bold_only_line_is_treated_as_a_heading() -> None:
    assert "heading" in kinds("Lede.\n\n**Upgrade path**\n\nDo the thing.")


def test_horizontal_rules_become_separators() -> None:
    assert "rule" in kinds("Lede.\n\n---\n\nMore.")


def test_an_empty_answer_yields_no_blocks() -> None:
    assert parse_blocks("") == []
    assert parse_blocks("Sources\n[1] x — https://x") == []


# ── inline markup ────────────────────────────────────────────────────────


def test_bold_and_code_become_pango_markup() -> None:
    markup = block_markup(parse_blocks("Use **npm** and `npx next`.")[0])
    assert "<b>npm</b>" in markup
    assert 'font_family="monospace"' in markup


def test_markup_special_characters_are_escaped() -> None:
    markup = block_markup(parse_blocks("Use a < b && c > d")[0])
    assert "&lt;" in markup and "&amp;" in markup
    assert "<b>" not in markup


def test_links_render_as_their_text_not_the_url() -> None:
    markup = block_markup(parse_blocks("See [the release notes](https://nextjs.org/blog).")[0])
    assert "the release notes" in markup
    assert "https://" not in markup


# ── headline extraction ──────────────────────────────────────────────────


def test_headline_is_the_first_sentence_without_markup() -> None:
    assert headline_text("**Next.js 16.2** is current [1].\n\n- a\n- b") == "Next.js 16.2 is current."


def test_headline_falls_back_to_the_first_bullet() -> None:
    assert headline_text("- first bullet\n- second") == "first bullet"


def test_headline_falls_back_to_the_first_pair() -> None:
    assert headline_text("- **Version** — 16.2\n- **Date** — today") == "Version — 16.2"


# ── the research fallback answer ─────────────────────────────────────────
#
# When synthesis fails, research_web answers with "Based on N sources:" and a
# run of "[n] excerpt…" lines. Those lines are the body, not the source list —
# stripping them as if they were sources left the answer card empty.

FALLBACK = """Based on 5 sources:

[1] Next.js 16.2 shipped with 400% faster dev startup and 200+ Turbopack fixes…
[2] It also adds new tooling for AI agents and better debugging…

Sources
[1] Next.js Blog — https://nextjs.org/blog
[2] LogRocket — https://blog.logrocket.com
"""


def test_the_research_fallback_keeps_its_body() -> None:
    body = strip_sources(FALLBACK)
    assert "400% faster dev startup" in body
    assert "AI agents" in body
    assert "nextjs.org" not in body
    assert "Based on 5 sources" not in body


def test_numbered_excerpts_stay_on_separate_lines() -> None:
    """A citation regex matching \\s* used to swallow the newline between them."""
    blocks = parse_blocks(FALLBACK)
    bullets = [b for b in blocks if b.kind == "bullets"]
    assert bullets and len(bullets[0].items) == 2


def test_a_source_entry_needs_a_url_to_be_stripped() -> None:
    assert strip_sources("[1] plain body text") == "[1] plain body text"
    assert strip_sources("Body.\n[1] Title — https://x") == "Body."


def test_the_fallback_answer_is_never_blank() -> None:
    for answer in (FALLBACK, "Based on 3 sources:\n[1] only this…", "plain answer"):
        assert strip_sources(answer).strip(), f"blanked: {answer!r}"


def test_only_a_sources_block_still_yields_nothing() -> None:
    assert strip_sources("Sources\n[1] x — https://x") == ""


# ── summary length ───────────────────────────────────────────────────────


def test_a_short_answer_is_shown_whole() -> None:
    from ui.canvas import is_compact

    assert is_compact("Next.js 16.2 is the current release.") is True


def test_a_long_answer_is_not_compact() -> None:
    from ui.canvas import is_compact

    assert is_compact("word " * 200) is False


def test_a_multi_item_list_is_not_compact() -> None:
    """Three 400-character excerpts used to render in full and fill the card."""
    from ui.canvas import is_compact

    assert is_compact(FALLBACK) is False


def test_the_summary_is_capped() -> None:
    from ui.canvas import SUMMARY_CHARS

    summary = headline_text(FALLBACK)
    assert 0 < len(summary) <= SUMMARY_CHARS


def test_the_summary_prefers_a_sentence_boundary() -> None:
    from ui.canvas import trim_to_sentence

    first = "Next.js 16.2 ships a much faster development server and a long list of fixes across the toolchain."
    assert trim_to_sentence(first + " " + "tail " * 60, 200) == first


def test_a_very_short_opening_sentence_is_not_used_as_the_whole_summary() -> None:
    """Cutting at a 20-character sentence would throw the answer away."""
    from ui.canvas import trim_to_sentence

    got = trim_to_sentence("Short lede. " + "detail " * 60, 200)
    assert len(got) > 100


def test_the_summary_falls_back_to_a_word_boundary() -> None:
    from ui.canvas import trim_to_sentence

    got = trim_to_sentence("alpha beta gamma delta epsilon zeta eta theta", 20)
    assert got.endswith("…") and " " in got and len(got) <= 21


def test_short_text_is_returned_unchanged() -> None:
    from ui.canvas import trim_to_sentence

    assert trim_to_sentence("Short.", 200) == "Short."
