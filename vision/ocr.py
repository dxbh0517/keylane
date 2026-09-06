"""Find a piece of text on screen, and say where it is.

This exists because of a limit, not an ambition. Keylane's vision model is a
4B one on an NPU, and asking it for pixel coordinates of a button produces
answers that are plausible and wrong — the ring lands near the target often
enough to look like it works and rarely enough to be useless.

OCR turns the same question into one that has an exact answer. "Point at the
Export button" stops being a grounding problem and becomes a text lookup with
a box attached. When tesseract is not installed the model's own coordinates
are used instead, and they are visibly worse; that is the honest trade and it
is why ``overlay.ocr_snap`` defaults on.

Parsing is separated from running tesseract so the matching logic can be
tested without the binary present.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Sparse-text mode. The default assumes a page of prose; a screenful of
# scattered buttons and menu items is exactly what --psm 11 is for.
PSM_SPARSE = "11"
_TIMEOUT = 20

# Below this, tesseract is guessing at noise.
MIN_CONFIDENCE = 40.0
# How many words on one line may be joined to match a multi-word target.
MAX_PHRASE_WORDS = 6

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Word:
    """One recognised word and the box it sits in, in image pixels."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    line: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class Match:
    """Where a phrase was found, in image pixels."""

    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        return self.left + self.width / 2, self.top + self.height / 2


def available() -> bool:
    return shutil.which("tesseract") is not None


def normalize(text: str) -> str:
    """Fold to what a human would call "the same words"."""
    return _SPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def parse_tsv(raw: str) -> list[Word]:
    """Words from tesseract's TSV output, dropping its structural rows.

    Rows with ``level`` below 5 describe pages, blocks and lines rather than
    words, and carry no text; a confidence of -1 marks a box tesseract found
    but could not read.
    """
    words: list[Word] = []
    lines = raw.splitlines()
    if not lines:
        return words

    header = lines[0].split("\t")
    try:
        index = {name: header.index(name) for name in
                 ("level", "left", "top", "width", "height", "conf", "text",
                  "block_num", "par_num", "line_num", "word_num")}
    except ValueError:
        logger.info("unexpected tesseract TSV header: %r", lines[0][:120])
        return words

    for row in lines[1:]:
        cells = row.split("\t")
        if len(cells) <= index["text"]:
            continue
        try:
            if int(cells[index["level"]]) != 5:
                continue
            confidence = float(cells[index["conf"]])
            left = int(cells[index["left"]])
            top = int(cells[index["top"]])
            width = int(cells[index["width"]])
            height = int(cells[index["height"]])
            line_key = (
                int(cells[index["block_num"]]),
                int(cells[index["par_num"]]),
                int(cells[index["line_num"]]),
                int(cells[index["word_num"]]),
            )
        except ValueError:
            continue
        text = cells[index["text"]].strip()
        if not text or confidence < 0:
            continue
        words.append(
            Word(
                text=text,
                left=left,
                top=top,
                width=width,
                height=height,
                confidence=confidence,
                line=line_key,
            )
        )
    return words


def _phrases(words: list[Word]) -> list[tuple[str, Match]]:
    """Every run of adjacent words on one line, as a searchable phrase.

    A target like "Save As…" is three tokens to tesseract and one label to the
    user, so single words alone would never match it.
    """
    out: list[tuple[str, Match]] = []
    for start in range(len(words)):
        block, par, line, word_no = words[start].line
        left = top = right = bottom = None
        pieces: list[str] = []
        total_confidence = 0.0
        for span in range(MAX_PHRASE_WORDS):
            index = start + span
            if index >= len(words):
                break
            word = words[index]
            block_no, par_no, line_no, word_no_here = word.line
            # Adjacency is structural: same line, consecutive word numbers.
            if (block_no, par_no, line_no) != (block, par, line) or word_no_here != word_no + span:
                break
            pieces.append(word.text)
            total_confidence += word.confidence
            left = word.left if left is None else min(left, word.left)
            top = word.top if top is None else min(top, word.top)
            right = word.right if right is None else max(right, word.right)
            bottom = word.bottom if bottom is None else max(bottom, word.bottom)
            phrase = normalize(" ".join(pieces))
            if phrase:
                out.append(
                    (
                        phrase,
                        Match(
                            text=" ".join(pieces),
                            left=int(left),
                            top=int(top),
                            width=int(right - left),
                            height=int(bottom - top),
                            confidence=total_confidence / len(pieces),
                        ),
                    )
                )
    return out


def find(words: list[Word], target: str) -> Match | None:
    """The best on-screen match for *target*, or None.

    Exact beats prefix beats substring, and within a tier the most confident
    reading wins. Ranking rather than first-hit matters: a toolbar with both
    "Export" and "Export as PDF" should resolve "Export" to the former.
    """
    wanted = normalize(target)
    if not wanted:
        return None

    tiers: list[tuple[int, float, Match]] = []
    for phrase, match in _phrases(words):
        if match.confidence < MIN_CONFIDENCE:
            continue
        if phrase == wanted:
            rank = 0
        elif phrase.startswith(wanted) or wanted.startswith(phrase):
            rank = 1
        elif wanted in phrase or phrase in wanted:
            rank = 2
        else:
            continue
        # Shorter phrases win ties: "Export" is a better hit for "Export"
        # than "Export as PDF" is.
        tiers.append((rank, -match.confidence + len(phrase) * 0.01, match))

    if not tiers:
        return None
    tiers.sort(key=lambda row: (row[0], row[1]))
    return tiers[0][2]


def read_words(image: bytes) -> list[Word]:
    """Run tesseract over a PNG. Returns [] if it is not installed or fails."""
    if not available():
        return []
    tmp = Path(tempfile.mkdtemp()) / "shot.png"
    try:
        tmp.write_bytes(image)
        done = subprocess.run(
            ["tesseract", str(tmp), "stdout", "--psm", PSM_SPARSE, "tsv"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.info("tesseract failed: %s", exc)
        return []
    finally:
        tmp.unlink(missing_ok=True)
        try:
            tmp.parent.rmdir()
        except OSError:
            pass

    if done.returncode != 0:
        logger.info("tesseract exited %d: %s", done.returncode, done.stderr[:200])
        return []
    return parse_tsv(done.stdout)


def locate(image: bytes, target: str) -> Match | None:
    """Where *target* is in *image*, in image pixels."""
    return find(read_words(image), target)
