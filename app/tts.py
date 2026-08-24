"""Text to speech — read an answer aloud, locally.

Keylane is local-first, so speech is too. Three engines are supported, in
descending order of quality:

``piper``       neural, natural-sounding, needs a downloaded ``.onnx`` voice
``espeak-ng``   formant synthesis, instant, 140+ languages, robotic
``flite``       small and fast, a handful of English voices

Engines are probed at runtime rather than assumed. The control panel lists
whatever this machine actually has, with the voices each one offers, and says
plainly what is missing and how to get it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Where Piper voices live. The first existing directory wins.
PIPER_VOICE_DIRS = [
    Path.home() / ".local/share/piper/voices",
    Path.home() / ".local/share/piper-voices",
    Path("/usr/share/piper-voices"),
    Path("/usr/share/piper/voices"),
]

MAX_SPEAK_CHARS = 8000

# Synthesis is fast; playback takes about as long as the speech itself. The
# budget is derived from the text so a long answer is not cut off, while a
# wedged player cannot hang the request forever.
SYNTH_TIMEOUT = 60
MIN_PLAY_TIMEOUT = 20
MAX_PLAY_TIMEOUT = 420
# Rough speaking rate: ~15 characters a second at normal speed.
CHARS_PER_SECOND = 15.0

# Markdown and shell noise that should not be read out loud.
_STRIP = [
    (re.compile(r"```.*?```", re.DOTALL), " code block "),
    (re.compile(r"`([^`]+)`"), r"\1"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])"), r"\1"),
    (re.compile(r"^#{1,6}\s*", re.MULTILINE), ""),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    (re.compile(r"^\s*[-•]\s*", re.MULTILINE), ", "),
    (re.compile(r"https?://\S+"), " a link "),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{2,}"), ". "),
]


class Voice(BaseModel):
    id: str
    name: str
    engine: str
    language: str = ""
    quality: str = ""


class EngineInfo(BaseModel):
    id: str
    name: str
    available: bool
    description: str = ""
    detail: str = ""
    install_hint: str = ""
    voices: list[Voice] = Field(default_factory=list)


@dataclass
class _Probe:
    """Cached engine discovery — probing shells out, so do it once."""

    engines: list[EngineInfo] = field(default_factory=list)
    stamp: float = 0.0


_cache = _Probe()
CACHE_SECONDS = 60.0


def clean_for_speech(text: str, limit: int = MAX_SPEAK_CHARS) -> str:
    """Strip markup so the synthesiser reads words, not punctuation."""
    out = text or ""
    for pattern, replacement in _STRIP:
        out = pattern.sub(replacement, out)
    out = out.strip()
    if len(out) > limit:
        # Cut at a sentence end so it does not stop mid-word.
        clipped = out[:limit]
        stop = max(clipped.rfind(". "), clipped.rfind("! "), clipped.rfind("? "))
        out = clipped[: stop + 1] if stop > limit // 2 else clipped
    return out


# ------------------------------------------------------------------ piper --


def _piper_voice_dir() -> Path | None:
    for directory in PIPER_VOICE_DIRS:
        if directory.is_dir():
            return directory
    return None


def _piper_voices() -> list[Voice]:
    directory = _piper_voice_dir()
    if directory is None:
        return []
    voices: list[Voice] = []
    for model in sorted(directory.rglob("*.onnx")):
        stem = model.stem  # en_GB-alba-medium
        parts = stem.split("-")
        language = parts[0] if parts else ""
        quality = parts[-1] if len(parts) > 2 else ""
        name = parts[1].replace("_", " ").title() if len(parts) > 1 else stem
        voices.append(
            Voice(
                id=str(model),
                name=f"{name} ({language})" if language else name,
                engine="piper",
                language=language,
                quality=quality,
            )
        )
    return voices


# --------------------------------------------------------------- espeak-ng --


def _espeak_voices() -> list[Voice]:
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if binary is None:
        return []
    try:
        output = subprocess.run(
            [binary, "--voices"], capture_output=True, text=True, timeout=8, check=False
        ).stdout
    except Exception:  # noqa: BLE001
        return []

    voices: list[Voice] = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        language, name = parts[1], parts[3]
        voices.append(
            Voice(
                id=language,
                name=f"{name.replace('_', ' ')} ({language})",
                engine="espeak",
                language=language,
            )
        )
    # 141 voices is a wall in a dropdown; lead with English, keep the rest.
    voices.sort(key=lambda v: (not v.language.startswith("en"), v.name.lower()))
    return voices


# ------------------------------------------------------------------ flite --


def _flite_voices() -> list[Voice]:
    binary = shutil.which("flite")
    if binary is None:
        return []
    try:
        output = subprocess.run(
            [binary, "-lv"], capture_output=True, text=True, timeout=8, check=False
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    names = output.replace("Voices available:", "").split()
    return [
        Voice(id=name, name=name.replace("_", " "), engine="flite", language="en")
        for name in names
        if name
    ]


# ------------------------------------------------------------- discovery --


def probe_engines(*, refresh: bool = False) -> list[EngineInfo]:
    """What this machine can actually speak with."""
    import time

    now = time.monotonic()
    if _cache.engines and not refresh and (now - _cache.stamp) < CACHE_SECONDS:
        return _cache.engines

    piper_bin = shutil.which("piper")
    piper_voices = _piper_voices() if piper_bin else []
    voice_dir = _piper_voice_dir()
    engines = [
        EngineInfo(
            id="piper",
            name="Piper",
            available=bool(piper_bin and piper_voices),
            description="Neural speech. The most natural of the three, still fully local.",
            detail=(
                f"{len(piper_voices)} voice{'s' if len(piper_voices) != 1 else ''} in {voice_dir}"
                if piper_voices
                else (
                    "piper is installed but no .onnx voices were found"
                    if piper_bin
                    else "piper is not installed"
                )
            ),
            install_hint=(
                ""
                if piper_voices
                else (
                    "Download a voice into ~/.local/share/piper/voices — for example "
                    "en_GB-alba-medium from huggingface.co/rhasspy/piper-voices"
                    if piper_bin
                    else "sudo dnf install piper, then download a voice into "
                    "~/.local/share/piper/voices"
                )
            ),
            voices=piper_voices,
        ),
        EngineInfo(
            id="espeak",
            name="eSpeak NG",
            available=bool(shutil.which("espeak-ng") or shutil.which("espeak")),
            description="Formant synthesis. Robotic, instant, and speaks 140+ languages.",
            detail="",
            install_hint="sudo dnf install espeak-ng",
            voices=_espeak_voices(),
        ),
        EngineInfo(
            id="flite",
            name="Flite",
            available=bool(shutil.which("flite")),
            description="Small and fast, with a handful of English voices.",
            detail="",
            install_hint="sudo dnf install flite",
            voices=_flite_voices(),
        ),
    ]
    for engine in engines:
        if engine.available and not engine.detail:
            engine.detail = f"{len(engine.voices)} voices"

    _cache.engines = engines
    _cache.stamp = now
    return engines


def default_engine() -> str:
    """The best engine present, preferring quality."""
    for engine in probe_engines():
        if engine.available:
            return engine.id
    return ""


def resolve(engine_id: str, voice_id: str) -> tuple[EngineInfo | None, str]:
    """Pick the engine and voice to use, falling back sensibly."""
    engines = {e.id: e for e in probe_engines()}
    engine = engines.get(engine_id)
    if engine is None or not engine.available:
        fallback = default_engine()
        engine = engines.get(fallback)
        voice_id = ""
    if engine is None:
        return None, ""
    if voice_id and any(v.id == voice_id for v in engine.voices):
        return engine, voice_id
    return engine, (engine.voices[0].id if engine.voices else "")


# ------------------------------------------------------------- synthesis --


class SpeakError(RuntimeError):
    pass


def _command(
    engine: EngineInfo, voice: str, rate: int, pitch: int, body: str
) -> list[str]:
    """Build the synthesiser invocation.

    Piper and eSpeak read stdin; flite does not — it needs the text as an
    argument, and silently emits an empty 44-byte WAV if you pipe to it.
    """
    if engine.id == "piper":
        binary = shutil.which("piper") or "piper"
        cmd = [binary, "--model", voice, "--output-raw"]
        if rate != 100:
            # Piper's length_scale is inverse: >1 is slower.
            cmd += ["--length-scale", f"{100 / max(rate, 25):.2f}"]
        return cmd
    if engine.id == "espeak":
        binary = shutil.which("espeak-ng") or shutil.which("espeak") or "espeak-ng"
        cmd = [binary, "--stdout", "-s", str(int(175 * rate / 100))]
        if voice:
            cmd += ["-v", voice]
        if pitch != 50:
            cmd += ["-p", str(max(0, min(pitch, 99)))]
        return cmd
    if engine.id == "flite":
        binary = shutil.which("flite") or "flite"
        cmd = [binary, "-t", body, "-o", "/dev/stdout"]
        if voice:
            cmd += ["-voice", voice]
        return cmd
    raise SpeakError(f"Unknown engine '{engine.id}'")


def _player(engine_id: str) -> list[str] | None:
    """How to play what the synthesiser writes to stdout."""
    # Piper emits raw 16-bit mono PCM at 22.05 kHz; the others emit WAV.
    if engine_id == "piper":
        for binary, args in (
            ("pw-cat", ["--playback", "--format", "s16", "--rate", "22050", "--channels", "1", "-"]),
            ("aplay", ["-r", "22050", "-f", "S16_LE", "-t", "raw", "-"]),
            ("paplay", ["--raw", "--format=s16le", "--rate=22050", "--channels=1"]),
        ):
            path = shutil.which(binary)
            if path:
                return [path, *args]
        return None
    for binary, args in (
        ("pw-play", ["-"]),
        ("paplay", ["-"]),
        ("aplay", ["-q", "-"]),
    ):
        path = shutil.which(binary)
        if path:
            return [path, *args]
    return None


async def speak(
    text: str,
    *,
    engine_id: str = "",
    voice_id: str = "",
    rate: int = 100,
    pitch: int = 50,
) -> dict[str, Any]:
    """Read text aloud. Returns what was actually used."""
    body = clean_for_speech(text)
    if not body:
        raise SpeakError("There is nothing to read.")

    engine, voice = resolve(engine_id, voice_id)
    if engine is None:
        raise SpeakError(
            "No speech engine is installed. Try: sudo dnf install piper espeak-ng"
        )
    if engine.id == "piper" and not voice:
        raise SpeakError(
            "Piper has no voices installed. Download one into "
            "~/.local/share/piper/voices."
        )

    player = _player(engine.id)
    if player is None:
        raise SpeakError("No audio player found. Try: sudo dnf install pipewire-utils")

    synth = _command(engine, voice, rate, pitch, body)
    # flite takes its text as an argument, so it gets no stdin at all.
    feeds_stdin = engine.id != "flite"
    logger.info("Speaking with %s (%s)", engine.id, voice or "default")

    # A real OS pipe between the two processes: asyncio's `stdout` is a
    # StreamReader, which cannot be passed as another process's stdin.
    read_fd, write_fd = os.pipe()
    try:
        try:
            producer = await asyncio.create_subprocess_exec(
                *synth,
                stdin=asyncio.subprocess.PIPE if feeds_stdin else asyncio.subprocess.DEVNULL,
                stdout=write_fd,
                stderr=asyncio.subprocess.PIPE,
            )
            consumer = await asyncio.create_subprocess_exec(
                *player,
                stdin=read_fd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        finally:
            # Both children hold their own copies now. Ours must go, or the
            # player never sees EOF and the call hangs forever.
            os.close(write_fd)
            os.close(read_fd)

        _out, synth_err = await asyncio.wait_for(
            producer.communicate(body.encode("utf-8") if feeds_stdin else None),
            timeout=SYNTH_TIMEOUT,
        )
        play_budget = min(
            max(len(body) / CHARS_PER_SECOND * (100 / max(rate, 25)) + 10, MIN_PLAY_TIMEOUT),
            MAX_PLAY_TIMEOUT,
        )
        await asyncio.wait_for(consumer.wait(), timeout=play_budget)
    except asyncio.TimeoutError as exc:
        for proc in (locals().get("producer"), locals().get("consumer")):
            if proc is not None and proc.returncode is None:
                proc.kill()
        raise SpeakError("Speech timed out.") from exc
    except FileNotFoundError as exc:
        raise SpeakError(f"{exc.filename} is not installed.") from exc

    if producer.returncode not in (0, None):
        detail = synth_err.decode("utf-8", "replace").strip().splitlines()
        raise SpeakError(
            f"{engine.name} failed: {detail[-1][:200] if detail else producer.returncode}"
        )

    return {
        "engine": engine.id,
        "voice": voice,
        "characters": len(body),
        "rate": rate,
    }


async def stop() -> int:
    """Silence anything currently speaking. Returns how many were stopped."""
    stopped = 0
    for name in ("piper", "espeak-ng", "espeak", "flite"):
        try:
            result = await asyncio.create_subprocess_exec(
                "pkill", "-f", f"^{shutil.which(name) or name}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await result.wait()
            if result.returncode == 0:
                stopped += 1
        except Exception:  # noqa: BLE001
            continue
    return stopped
