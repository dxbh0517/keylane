"""Audio8 speech: chunking, voice references, and honest degradation.

The model itself is 1.7 GB and slow on CPU, so nothing here loads it. What is
tested is everything around it — the parts that decide *what* gets synthesised
and what happens when it cannot be.
"""

from __future__ import annotations

import pytest

from app import tts
from app.tts import (
    MAX_CHUNK_CHARS,
    SpeakError,
    clean_for_speech,
    probe_engines,
    resolve,
    split_for_synthesis,
)


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    tts._cache.engines = []
    tts._cache.stamp = 0.0
    yield
    tts._cache.engines = []
    tts._cache.stamp = 0.0


# ------------------------------------------------------------------ chunking


def test_short_text_is_one_chunk():
    assert split_for_synthesis("Hello there.") == ["Hello there."]


def test_text_is_split_on_sentence_boundaries():
    body = " ".join(f"This is sentence number {n}." for n in range(1, 30))

    chunks = split_for_synthesis(body)

    assert len(chunks) > 1
    # Every chunk must fit the generate() budget, or it comes back truncated.
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    # Splitting must not lose or reorder words.
    assert " ".join(chunks).split() == body.split()


def test_a_sentence_longer_than_the_limit_is_broken_on_whitespace():
    body = "word " * 200

    chunks = split_for_synthesis(body.strip())

    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    # Never mid-word: every piece is whole words.
    assert all(not c.startswith(" ") and "wor d" not in c for c in chunks)
    assert " ".join(chunks).split() == body.split()


def test_empty_text_yields_no_chunks():
    assert split_for_synthesis("   ") == []


def test_markup_is_stripped_before_chunking():
    cleaned = clean_for_speech("**Disk** is `74%` full.")
    assert "*" not in cleaned and "`" not in cleaned


# -------------------------------------------------------------------- voices


def test_a_clip_without_a_transcript_is_not_offered(tmp_path, monkeypatch):
    # Cloning conditions on the transcript; guessing it degrades the voice
    # badly, so a clip without one is skipped rather than half-used.
    (tmp_path / "alba.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(tts, "voices_dir", lambda: tmp_path)

    assert tts._reference_voices() == []


def test_a_clip_with_a_transcript_becomes_a_voice(tmp_path, monkeypatch):
    (tmp_path / "alba.wav").write_bytes(b"RIFF")
    (tmp_path / "alba.txt").write_text("This is what the clip says.", encoding="utf-8")
    monkeypatch.setattr(tts, "voices_dir", lambda: tmp_path)

    voices = tts._reference_voices()

    assert [v.name for v in voices] == ["Alba"]
    assert voices[0].id == str(tmp_path / "alba.wav")


def test_the_default_voice_needs_no_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(tts, "voices_dir", lambda: tmp_path)

    voices = tts._voices()

    assert voices[0].id == ""
    assert tts._reference_for("") is None


def test_a_reference_carries_its_transcript(tmp_path, monkeypatch):
    (tmp_path / "alba.wav").write_bytes(b"RIFF")
    (tmp_path / "alba.txt").write_text("Spoken words.", encoding="utf-8")
    monkeypatch.setattr(tts, "voices_dir", lambda: tmp_path)

    reference = tts._reference_for(str(tmp_path / "alba.wav"))

    assert reference == (str(tmp_path / "alba.wav"), "Spoken words.")


def test_an_empty_transcript_is_treated_as_missing(tmp_path):
    (tmp_path / "alba.wav").write_bytes(b"RIFF")
    (tmp_path / "alba.txt").write_text("   ", encoding="utf-8")

    assert tts._reference_for(str(tmp_path / "alba.wav")) is None


# --------------------------------------------------------------- degradation


def test_the_engine_says_what_is_missing(monkeypatch):
    monkeypatch.setattr(tts, "model_installed", lambda: False)
    monkeypatch.setattr(tts, "_missing_dependency", lambda: "")

    engine = probe_engines(refresh=True)[0]

    assert engine.available is False
    assert "not been downloaded" in engine.detail
    assert engine.install_hint, "an unavailable engine must say how to fix it"


def test_a_missing_python_package_is_named(monkeypatch):
    monkeypatch.setattr(tts, "model_installed", lambda: True)
    monkeypatch.setattr(tts, "_missing_dependency", lambda: "soundfile")

    engine = probe_engines(refresh=True)[0]

    assert engine.available is False
    assert "soundfile" in engine.install_hint


def test_settings_naming_a_retired_engine_do_not_strand_the_user(monkeypatch):
    # Anyone upgrading has "piper" or "espeak" saved in assistant.toml.
    monkeypatch.setattr(tts, "model_installed", lambda: True)
    monkeypatch.setattr(tts, "_missing_dependency", lambda: "")

    engine, voice = resolve("piper", "/old/path/en_GB-alba-medium.onnx")

    assert engine is not None and engine.id == "audio8"
    assert voice == "", "a piper voice path is meaningless to Audio8"


@pytest.mark.asyncio
async def test_speaking_without_the_model_explains_rather_than_crashes(monkeypatch):
    monkeypatch.setattr(tts, "model_installed", lambda: False)
    monkeypatch.setattr(tts, "_missing_dependency", lambda: "")

    with pytest.raises(SpeakError) as caught:
        await tts.speak("hello")

    assert "not installed" in str(caught.value)


@pytest.mark.asyncio
async def test_speaking_nothing_is_refused():
    with pytest.raises(SpeakError):
        await tts.speak("   ")


@pytest.mark.asyncio
async def test_stop_is_safe_when_nothing_is_playing():
    assert await tts.stop() == 0


def test_rate_and_pitch_are_advertised_as_unsupported(monkeypatch):
    # The panel offers sliders; it must be able to tell they would do nothing.
    monkeypatch.setattr(tts, "model_installed", lambda: True)
    monkeypatch.setattr(tts, "_missing_dependency", lambda: "")

    engine = probe_engines(refresh=True)[0]

    assert engine.supports_rate is False
    assert engine.supports_pitch is False


@pytest.mark.asyncio
async def test_stop_cancels_a_synthesis_still_in_flight(monkeypatch, tmp_path):
    """Synthesis is the long part, and it cannot be interrupted mid-thread.

    Stop must still mean stop: audio generated after the user gave up is
    thrown away rather than spoken at them a minute late.
    """
    monkeypatch.setattr(tts, "model_installed", lambda: True)
    monkeypatch.setattr(tts, "_missing_dependency", lambda: "")
    monkeypatch.setattr(tts, "_player", lambda: ["/bin/true"])

    async def slow_render(text, reference):
        # Stand in for a minutes-long generate(): the user gives up part way.
        await tts.stop()
        return b"RIFFfake"

    def blocking_render(text, reference):
        raise AssertionError("should not be reached")

    monkeypatch.setattr(tts, "_render_wav", blocking_render)

    async def fake_to_thread(fn, *args, **kwargs):
        return await slow_render(*args)

    monkeypatch.setattr(tts.asyncio, "to_thread", fake_to_thread)

    result = await tts.speak("some words to read")

    assert result.get("stopped") is True
