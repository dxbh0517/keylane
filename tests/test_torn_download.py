"""A download that stopped halfway, and the error it used to produce.

Selecting a model seven seconds into a 4.6 GB download probed a directory that
was still filling. `missing_weights` checked only that *some* tokenizer file
existed, which a half-written one satisfies, so the probe went ahead and the
tokenizer failed from the inside with

    Invalid range in '{}' in regular expression

— which names nothing to act on and does not look like a download problem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtimes import backend_for
from runtimes.probe_runner import last_line


def _export(root: Path, *, tokenizer: bytes | None = b"x" * 4096, merges: bytes | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "genai_config.json").write_text(
        json.dumps({"model": {"context_length": 4096, "decoder": {"filename": "model.onnx"}}}),
        encoding="utf-8",
    )
    (root / "model.onnx").write_bytes(b"\0" * 16384)
    (root / "model.onnx.data").write_bytes(b"\0" * 16384)
    if tokenizer is not None:
        (root / "tokenizer.json").write_bytes(tokenizer)
    if merges is not None:
        (root / "vocab.json").write_bytes(b"y" * 4096)
        (root / "merges.txt").write_bytes(merges)
    return root


def test_a_complete_export_is_complete(tmp_path: Path) -> None:
    assert backend_for("onnxruntime").missing_weights(_export(tmp_path / "m")) == []


def test_a_half_written_tokenizer_is_caught(tmp_path: Path) -> None:
    """Existence was the check, and existence is what a torn file satisfies."""
    missing = backend_for("onnxruntime").missing_weights(
        _export(tmp_path / "m", tokenizer=b"{")
    )
    assert missing == ["tokenizer.json (empty or truncated)"]


def test_an_absent_tokenizer_is_caught(tmp_path: Path) -> None:
    assert backend_for("onnxruntime").missing_weights(
        _export(tmp_path / "m", tokenizer=None)
    ) == ["tokenizer.json"]


def test_a_bpe_pair_needs_both_halves(tmp_path: Path) -> None:
    """vocab.json without merges.txt is a merge table that cannot be built."""
    root = _export(tmp_path / "m", merges=b"z" * 4096)
    (root / "merges.txt").unlink()
    assert "merges.txt" in backend_for("onnxruntime").missing_weights(root)


def test_a_half_written_merges_file_is_caught(tmp_path: Path) -> None:
    missing = backend_for("onnxruntime").missing_weights(
        _export(tmp_path / "m", merges=b"a")
    )
    assert "merges.txt (empty or truncated)" in missing


# ── the message has to survive the subprocess boundary ───────────────────


def test_a_multi_line_failure_keeps_the_part_that_helps() -> None:
    """The probe runs in a child, so its message crosses as text.

    Reading from the end returned the last fragment — "Underlying error: …" —
    and discarded the sentence naming the model, the VRAM it needed and what
    was holding the card.
    """
    traceback = (
        "Traceback (most recent call last):\n"
        '  File "/x/y.py", line 3, in <module>\n'
        "    raise RuntimeError(msg)\n"
        "RuntimeError: not enough free VRAM — phi-4-cuda needs about 9.0 GB "
        "and 2141 MiB is free. Close whatever else is using the GPU.\n"
        "\n"
        "Underlying error: bfc_arena.cc:359 Failed to allocate memory\n"
    )
    message = last_line(traceback)
    assert message.startswith("RuntimeError: not enough free VRAM")
    assert "2141 MiB is free" in message
    # The original is still carried, just no longer instead of the explanation.
    assert "bfc_arena" in message


def test_a_single_line_exception_is_unchanged() -> None:
    assert last_line(
        'Traceback (most recent call last):\n  File "a", line 1\nValueError: bad thing'
    ) == "ValueError: bad thing"


@pytest.mark.parametrize("text", ["", "   \n  \n"])
def test_nothing_at_all_is_not_a_crash(text: str) -> None:
    assert last_line(text) == ""


def test_output_that_is_not_a_traceback_is_passed_through() -> None:
    assert last_line("some tool wrote this to stderr") == "some tool wrote this to stderr"
