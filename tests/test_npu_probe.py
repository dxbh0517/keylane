"""Tests for the model probe that keeps a bad model from killing the gateway."""

from __future__ import annotations

import sys
import textwrap

import pytest

from app.npu import probe as probe_mod
from app.npu.probe import (
    cache_dir,
    failure_detail,
    failure_kind,
    probe,
    static_objection,
)

# ------------------------------------------------------------- static checks


def _write_ir(path, name: str, version: str) -> None:
    """A minimal IR stub carrying just the rt_info the probe reads."""
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(
        textwrap.dedent(
            f"""\
            <?xml version="1.0"?>
            <net name="tokenizer" version="11">
              <rt_info>
                <openvino_version value="{version}" />
              </rt_info>
            </net>
            """
        ),
        encoding="utf-8",
    )


def test_tokenizer_from_a_newer_openvino_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_mod, "_runtime_version", lambda: (2026, 2))
    _write_ir(tmp_path, "openvino_tokenizer.xml", "2026.4.0-22768-e97007fc748")
    (tmp_path / "openvino_model.xml").write_text("<net/>", encoding="utf-8")

    objection = static_objection(tmp_path)

    assert objection is not None
    assert "2026.4" in objection and "2026.2" in objection


def test_tokenizer_from_an_older_openvino_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_mod, "_runtime_version", lambda: (2026, 2))
    _write_ir(tmp_path, "openvino_tokenizer.xml", "2024.5.0")
    (tmp_path / "openvino_model.xml").write_text("<net/>", encoding="utf-8")

    assert static_objection(tmp_path) is None


def test_vision_language_export_is_refused_for_the_text_pipeline(tmp_path):
    # A VLM export has no openvino_model.xml — the language half is split out
    # and takes inputs_embeds, which LLMPipeline cannot feed.
    (tmp_path / "openvino_language_model.xml").write_text("<net/>", encoding="utf-8")

    objection = static_objection(tmp_path)

    assert objection is not None
    assert "vision-language" in objection


def test_model_with_a_vision_tower_is_refused(tmp_path):
    (tmp_path / "openvino_model.xml").write_text("<net/>", encoding="utf-8")
    (tmp_path / "openvino_vision_embeddings_model.xml").write_text("<net/>", encoding="utf-8")

    objection = static_objection(tmp_path)

    assert objection is not None
    assert "vision tower" in objection


def test_an_ordinary_text_model_raises_no_objection(tmp_path):
    (tmp_path / "openvino_model.xml").write_text("<net/>", encoding="utf-8")

    assert static_objection(tmp_path) is None


# -------------------------------------------------------------- the subprocess


def _probe_with(monkeypatch, source: str, **kwargs):
    """Run the probe against a stand-in script instead of OpenVINO."""
    monkeypatch.setattr(probe_mod, "_PROBE_SOURCE", source)
    return probe(kwargs.pop("path", "/nonexistent"), "NPU", cache=None, **kwargs)


def test_probe_reports_success(monkeypatch):
    ok, reason = _probe_with(monkeypatch, 'print("KEYLANE_PROBE_OK")')

    assert ok is True
    assert reason == "ready"


def test_a_segfaulting_load_is_reported_as_a_crash_not_an_exception(monkeypatch):
    # The case that used to take the whole gateway down: OpenVINO dying by
    # signal, which no try/except in the parent could ever catch.
    ok, reason = _probe_with(
        monkeypatch,
        "import ctypes; ctypes.string_at(0)",
    )

    assert ok is False
    assert failure_kind(reason) == "crash"
    assert "SIGSEGV" in failure_detail(reason)


def test_an_aborting_load_is_reported_as_a_crash(monkeypatch):
    ok, reason = _probe_with(monkeypatch, "import os, signal; os.kill(os.getpid(), signal.SIGABRT)")

    assert ok is False
    assert failure_kind(reason) == "crash"
    assert "SIGABRT" in failure_detail(reason)


def test_an_ordinary_exception_is_reported_as_an_error(monkeypatch):
    ok, reason = _probe_with(monkeypatch, 'raise RuntimeError("no such device")')

    assert ok is False
    assert failure_kind(reason) == "error"
    assert "no such device" in failure_detail(reason)


def test_a_slow_compile_is_reported_as_a_timeout(monkeypatch):
    ok, reason = _probe_with(monkeypatch, "import time; time.sleep(30)", timeout=1.0)

    assert ok is False
    assert failure_kind(reason) == "timeout"


def test_probe_never_raises_when_the_interpreter_cannot_start(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")

    ok, reason = probe("/nonexistent", "NPU", cache=None)

    assert ok is False
    assert failure_kind(reason) == "error"


# ---------------------------------------------------------------- cache dir


def test_cache_dir_is_created_and_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_mod, "_CACHE_OVERRIDE", str(tmp_path / "blobs"))

    first = cache_dir()
    second = cache_dir()

    assert first.is_dir()
    assert first == second


def test_cache_dir_defaults_under_the_install_root(tmp_path, monkeypatch):
    monkeypatch.setattr(probe_mod, "_CACHE_OVERRIDE", "")

    class FakeConfig:
        root = tmp_path

    path = cache_dir(FakeConfig())

    assert path == tmp_path / "cache" / "openvino"
    assert path.is_dir()
