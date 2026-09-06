"""Two small model calls that produce text to be typed somewhere else.

Both are deliberately *not* agent turns. A dictated sentence must not be able
to write a memory, schedule a task, or run a tool — it is going straight into
the user's editor, and the only thing that should come back is a string.

**Cleanup** fixes punctuation and casing on a transcript. The hard part is not
the prompt, it is refusing the model's improvements: a small model asked to
"clean up" text will happily rewrite it, and the user then watches their own
sentence get paraphrased. :func:`clean_transcript` therefore checks the result
against the input and discards it when the words moved.

**Compose** is the screen-aware one: the user says what they want, Keylane
looks at the screen, and the model writes the actual text.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

CLEANUP_SYSTEM = (
    "You repair speech-to-text output. Fix capitalisation, punctuation and "
    "obvious homophone slips. Never reword, never summarise, never translate, "
    "never answer what the text says. Reply with the corrected text alone."
)

COMPOSE_SYSTEM = (
    "You write text that will be typed directly into the user's application. "
    "Reply with that text and nothing else — no preamble, no explanation, no "
    "quotation marks around it, no markdown fences."
)

# A cleanup pass may add punctuation and change case. It may not change which
# words are there. Anything beyond this much drift is a rewrite.
_MAX_WORD_DRIFT = 0.25
_MAX_CLEANUP_TOKENS = 400
_MAX_COMPOSE_TOKENS = 600

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_FENCE = re.compile(r"^```[a-zA-Z0-9]*\n(.*)\n```$", re.DOTALL)
# Apostrophes are deleted rather than treated as separators before words are
# counted. Restoring one is the single most common thing a cleanup pass does,
# and splitting "Let's" into two tokens would score that correction as a
# rewrite and throw the whole corrected sentence away.
_APOSTROPHE = re.compile(r"['’ʼ]")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(_APOSTROPHE.sub("", text))]


def drifted(original: str, cleaned: str) -> bool:
    """Whether *cleaned* changed the words rather than only the punctuation."""
    before, after = _words(original), _words(cleaned)
    if not before:
        return bool(after)
    if not after:
        return True
    # Multiset difference in both directions: a model that drops half the
    # sentence and one that pads it are the same failure.
    counts: dict[str, int] = {}
    for word in before:
        counts[word] = counts.get(word, 0) + 1
    changed = 0
    for word in after:
        if counts.get(word, 0) > 0:
            counts[word] -= 1
        else:
            changed += 1
    changed += sum(counts.values())
    return changed / max(len(before), len(after)) > _MAX_WORD_DRIFT


def strip_wrapper(text: str) -> str:
    """Undo the packaging a small model adds around text it was asked for."""
    cleaned = text.strip()
    fenced = _FENCE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    return cleaned


def clean_transcript(text: str) -> str:
    """Punctuate and capitalise *text*, or return it untouched.

    Returning the input is a success, not a failure: the raw transcript is
    still exactly what the user said.
    """
    original = text.strip()
    if not original:
        return ""

    from seams import get_context

    try:
        raw = get_context().llm.generate(
            f"Correct this transcript:\n\n{original}",
            route="utility",
            system=CLEANUP_SYSTEM,
            max_new_tokens=_MAX_CLEANUP_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("transcript cleanup failed (%s); keeping the raw text", exc)
        return original

    from npu.thinking import extract_user_answer

    cleaned = strip_wrapper(extract_user_answer(raw))
    if not cleaned:
        return original
    if drifted(original, cleaned):
        logger.info("discarding cleanup: it reworded the transcript rather than punctuating it")
        return original
    return cleaned


def compose_text(instruction: str, app: str = "", image: bytes | None = None) -> str:
    """Draft the text the user asked for, given what is on their screen."""
    said = instruction.strip()
    if not said:
        return ""

    from seams import get_context

    where = f"The user is working in {app}. " if app else ""
    looking = (
        "A screenshot of their screen is attached; read it for the context they mean. "
        if image
        else ""
    )
    prompt = (
        f"{where}{looking}They asked you to write the following for them:\n\n{said}\n\n"
        "Write only the text to insert."
    )

    # `background` rather than `interactive`: a draft is worth the GPU when one
    # is configured, and nothing is blocking the HUD on it.
    raw = get_context().llm.generate(
        prompt,
        route="background",
        system=COMPOSE_SYSTEM,
        max_new_tokens=_MAX_COMPOSE_TOKENS,
        images=[image] if image else None,
    )

    from npu.thinking import extract_user_answer

    return strip_wrapper(extract_user_answer(raw))
