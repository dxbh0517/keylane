"""Themes: every built-in one fills the stylesheet, and a user theme patches it."""

from __future__ import annotations

import re

import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from ui.themes import (
    BUILTIN_DIR,
    DEFAULT_THEME_ID,
    ThemeError,
    list_themes,
    load_theme,
    missing_tokens,
    render_css,
    template_tokens,
)

BUILTIN_IDS = (
    "glass-console",
    "paper-terminal",
    "aurora",
    "copper-oxide",
    "nord-frost",
    "high-contrast",
)


@pytest.fixture()
def user_themes(tmp_path, monkeypatch):
    """A themes directory of our own, so the developer's own themes stay out."""
    monkeypatch.setattr("ui.themes.user_dir", lambda: tmp_path)
    return tmp_path


# ── the built-in themes ──────────────────────────────────────────────────


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_a_built_in_theme_fills_every_hole(theme_id: str, user_themes) -> None:
    theme = load_theme(theme_id)
    assert missing_tokens(theme) == []
    assert "{{" not in render_css(theme)


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
def test_a_built_in_theme_is_complete_without_inheriting(theme_id: str, user_themes) -> None:
    """Built-ins are the files people copy from; none may rely on a base."""
    with (BUILTIN_DIR / f"{theme_id}.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    assert "extends" not in raw.get("theme", {})
    assert set(raw["light"]) == set(raw["dark"])


def test_both_schemes_carry_the_same_tokens(user_themes) -> None:
    for theme_id in BUILTIN_IDS:
        theme = load_theme(theme_id)
        assert set(theme.light) == set(theme.dark), theme_id


def test_the_orb_colours_are_themeable(user_themes) -> None:
    """The orb paints itself, so its ramp has to come from the theme."""
    for theme_id in BUILTIN_IDS:
        theme = load_theme(theme_id)
        for scheme in (theme.light, theme.dark):
            assert scheme["orb-accent"].startswith("#")
            assert scheme["orb-accent-alt"].startswith("#")


def test_every_theme_lists(user_themes) -> None:
    ids = {t.id for t in list_themes()}
    assert set(BUILTIN_IDS) <= ids


# ── themes of your own ───────────────────────────────────────────────────


def test_a_user_theme_only_writes_what_it_changes(user_themes) -> None:
    (user_themes / "midnight.toml").write_text(
        '[theme]\nname = "Midnight"\n\n[dark]\naccent = "x"\npanel-bg = "#000000"\n'
    )
    theme = load_theme("midnight")
    assert theme.source == "user"
    assert theme.dark["panel-bg"] == "#000000"
    # everything it did not mention still resolves
    assert missing_tokens(theme) == []
    assert theme.light["panel-bg"] == load_theme(DEFAULT_THEME_ID).light["panel-bg"]


def test_a_user_theme_can_start_from_another(user_themes) -> None:
    (user_themes / "papercut.toml").write_text(
        '[theme]\nname = "Papercut"\nextends = "paper-terminal"\n\n[light]\npanel-bg = "#fff000"\n'
    )
    theme = load_theme("papercut")
    assert theme.light["panel-bg"] == "#fff000"
    assert theme.common["radius-panel"] == load_theme("paper-terminal").common["radius-panel"]


def test_a_user_theme_shadows_a_built_in_id(user_themes) -> None:
    (user_themes / "aurora.toml").write_text('[theme]\nname = "Mine"\n\n[dark]\npanel-bg = "#010203"\n')
    assert load_theme("aurora").dark["panel-bg"] == "#010203"
    assert load_theme("aurora").source == "user"


def test_a_theme_that_extends_itself_is_refused(user_themes) -> None:
    (user_themes / "loop.toml").write_text('[theme]\nname = "Loop"\nextends = "loop"\n')
    with pytest.raises(ThemeError):
        load_theme("loop")


def test_an_unreadable_theme_is_skipped_not_fatal(user_themes) -> None:
    (user_themes / "broken.toml").write_text("this is not toml = = =\n")
    assert {t.id for t in list_themes()} >= set(BUILTIN_IDS)


def test_a_missing_theme_says_so(user_themes) -> None:
    with pytest.raises(ThemeError):
        load_theme("no-such-theme")


# ── the template's side of the contract ──────────────────────────────────


def test_a_theme_missing_a_token_will_not_render(user_themes) -> None:
    scheme_names, _ = template_tokens()
    theme = load_theme("glass-console")
    theme.light.pop(sorted(scheme_names)[0])
    with pytest.raises(ThemeError):
        render_css(theme)


def test_both_schemes_are_rendered_into_one_stylesheet(user_themes) -> None:
    """Light/dark stays a class swap, so no reload on a scheme change."""
    css = render_css(load_theme("glass-console"))
    assert "window.spotlight-window.style-light .spotlight-panel" in css
    assert "window.spotlight-window.style-dark .spotlight-panel" in css


# ── contrast ─────────────────────────────────────────────────────────────
#
# Nothing checked whether a theme's text could actually be read on its own
# background. Most of the shipped themes are careful about it by eye; the point
# of high-contrast is to guarantee it, and a guarantee nobody measures is a
# hope.


def _rgb(value: str) -> tuple[float, float, float] | None:
    """Parse a theme colour. Returns None for anything that is not one.

    A handful of tokens hold a whole CSS value — a shadow, a gradient — and a
    few colours are rgba() over an unknown backdrop, which has no single
    contrast to measure. Those are skipped rather than guessed at.
    """
    text = value.strip()
    if text.startswith("#") and len(text) == 7:
        return tuple(int(text[i : i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]
    match = re.fullmatch(r"rgb\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)\s*\)", text)
    if match:
        return tuple(float(g) / 255 for g in match.groups())  # type: ignore[return-value]
    return None


def _luminance(rgb: tuple[float, float, float]) -> float:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str, bg: str) -> float | None:
    a, b = _rgb(fg), _rgb(bg)
    if a is None or b is None:
        return None
    la, lb = _luminance(a), _luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


# (text token, the surface it sits on)
TEXT_ON_SURFACE = (
    ("hud-text", "hud-bg"),
    ("hud-headline", "hud-bg"),
    ("hud-title", "hud-bg"),
    ("search-text", "panel-bg"),
    ("title-text", "window-bg"),
    ("section-title", "window-bg"),
    ("field-label", "window-bg"),
    ("control-text", "control-bg"),
    ("entry-text", "entry-bg"),
    ("nav-text", "nav-bg"),
    ("btn-primary-text", "btn-primary-bg"),
)


@pytest.mark.parametrize("theme_id", BUILTIN_IDS)
@pytest.mark.parametrize(("fg", "bg"), TEXT_ON_SURFACE)
def test_text_is_readable_on_its_own_surface(theme_id: str, fg: str, bg: str, user_themes) -> None:
    """WCAG AA for body text, on both schemes, for every shipped theme."""
    theme = load_theme(theme_id)
    for scheme_name, dark in (("light", False), ("dark", True)):
        tokens = theme.scheme(dark)
        ratio = contrast_ratio(tokens[fg], tokens[bg])
        if ratio is None:
            continue  # translucent over an unknown backdrop; nothing to measure
        assert ratio >= 4.5, (
            f"{theme_id} {scheme_name}: {fg} on {bg} is {ratio:.1f}:1, below AA"
        )


def test_high_contrast_earns_its_name(user_themes) -> None:
    """The whole reason this theme exists: AAA, not merely AA."""
    theme = load_theme("high-contrast")
    for dark in (False, True):
        tokens = theme.scheme(dark)
        for fg, bg in TEXT_ON_SURFACE:
            ratio = contrast_ratio(tokens[fg], tokens[bg])
            assert ratio is not None, f"{fg}/{bg} must be an opaque colour here"
            assert ratio >= 7.0, f"{fg} on {bg} is {ratio:.1f}:1, below AAA"


def test_high_contrast_uses_no_translucency(user_themes) -> None:
    """A translucent surface has whatever contrast the desktop behind it gives."""
    theme = load_theme("high-contrast")
    opaque_surfaces = (
        "panel-bg", "hud-bg", "window-bg", "nav-bg",
        "control-bg", "entry-bg", "popover-bg", "hud-card-bg",
    )
    for dark in (False, True):
        tokens = theme.scheme(dark)
        for token in opaque_surfaces:
            assert "rgba" not in tokens[token], f"{token} is translucent: {tokens[token]}"
