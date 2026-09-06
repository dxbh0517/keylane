"""Finding text on screen: TSV parsing and match ranking.

None of this needs tesseract installed — the parser is separated from the
subprocess precisely so the ranking can be tested, since the ranking is the
part that decides whether a ring lands on the right button.
"""

from __future__ import annotations

from vision.ocr import MIN_CONFIDENCE, Word, find, normalize, parse_tsv

HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
    "left\ttop\twidth\theight\tconf\ttext"
)


def _row(
    text: str,
    *,
    left: int = 0,
    top: int = 0,
    width: int = 50,
    height: int = 12,
    conf: float = 95.0,
    level: int = 5,
    block: int = 1,
    par: int = 1,
    line: int = 1,
    word: int = 1,
) -> str:
    return (
        f"{level}\t1\t{block}\t{par}\t{line}\t{word}\t"
        f"{left}\t{top}\t{width}\t{height}\t{conf}\t{text}"
    )


def _tsv(*rows: str) -> str:
    return "\n".join([HEADER, *rows])


# ── parsing ──────────────────────────────────────────────────────────────


def test_word_rows_are_read() -> None:
    words = parse_tsv(_tsv(_row("Export", left=100, top=40)))
    assert len(words) == 1
    assert words[0].text == "Export"
    assert (words[0].left, words[0].top) == (100, 40)


def test_structural_rows_are_ignored() -> None:
    """Levels below 5 describe pages and lines, and carry no text."""
    words = parse_tsv(_tsv(_row("", level=1, conf=-1), _row("Export")))
    assert [w.text for w in words] == ["Export"]


def test_boxes_tesseract_could_not_read_are_dropped() -> None:
    assert parse_tsv(_tsv(_row("", conf=-1))) == []


def test_a_missing_header_is_not_a_crash() -> None:
    assert parse_tsv("garbage\nmore garbage") == []


def test_empty_output_is_not_a_crash() -> None:
    assert parse_tsv("") == []


def test_normalize_folds_punctuation_and_case() -> None:
    assert normalize("Save As…") == "save as"


# ── ranking ──────────────────────────────────────────────────────────────


def _words(*specs) -> list[Word]:
    out = []
    for index, (text, left) in enumerate(specs, start=1):
        out.append(
            Word(
                text=text,
                left=left,
                top=10,
                width=40,
                height=12,
                confidence=95.0,
                line=(1, 1, 1, index),
            )
        )
    return out


def test_an_exact_match_beats_a_longer_one_containing_it() -> None:
    """A toolbar with both "Export" and "Export as PDF" must resolve "Export"."""
    words = _words(("Export", 100), ("as", 150), ("PDF", 180))
    match = find(words, "Export")
    assert match is not None
    assert match.text == "Export"


def test_a_multi_word_target_matches_across_adjacent_words() -> None:
    words = _words(("Save", 10), ("As", 60))
    match = find(words, "Save As")
    assert match is not None
    assert match.left == 10
    # The box spans both words, not just the first.
    assert match.width >= 90


def test_words_on_different_lines_are_not_joined_into_one_box() -> None:
    """A match may still land on "Save" alone — it must not span both rows.

    A box stretched over two lines would put the ring in the gap between them,
    pointing at neither.
    """
    words = [
        Word(text="Save", left=10, top=10, width=40, height=12, confidence=95.0, line=(1, 1, 1, 1)),
        Word(text="As", left=10, top=40, width=40, height=12, confidence=95.0, line=(1, 1, 2, 1)),
    ]
    match = find(words, "Save As")
    assert match is not None
    assert match.text == "Save"
    assert match.height == 12


def test_a_low_confidence_reading_is_not_pointed_at() -> None:
    words = [
        Word(
            text="Export",
            left=10,
            top=10,
            width=40,
            height=12,
            confidence=MIN_CONFIDENCE - 5,
            line=(1, 1, 1, 1),
        )
    ]
    assert find(words, "Export") is None


def test_matching_ignores_case_and_punctuation() -> None:
    assert find(_words(("EXPORT…", 10)), "export") is not None


def test_a_target_that_is_not_there_returns_nothing() -> None:
    """Answering "not found" is what lets the model say so instead of guessing."""
    assert find(_words(("Cancel", 10)), "Export") is None


def test_an_empty_target_matches_nothing() -> None:
    assert find(_words(("Export", 10)), "   ") is None


def test_the_match_centre_is_the_middle_of_its_box() -> None:
    match = find(_words(("Export", 100)), "Export")
    assert match is not None
    assert match.center == (120.0, 16.0)
