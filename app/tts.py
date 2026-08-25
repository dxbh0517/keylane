"""Text to speech — read an answer aloud, locally.

Speech is Audio8 TTS (``Audio8/Audio8-TTS-Preview-0.1b``): a 0.1B neural model
that runs in this process on torch. Unlike the shell-out synthesisers it
replaces, it produces audio as a numpy array rather than a stream on stdout, so
the shape here is: synthesise on a worker thread, then hand a WAV to the
system player.

Two consequences drive the rest of this module.

Generation is bounded by ``max_new_tokens``, so a long answer cannot be
synthesised in one pass — text is split on sentence boundaries and the pieces
are concatenated. And the model is zero-shot: a "voice" is a reference clip to
clone rather than a name the engine already knows, so voices are discovered
from a directory of wav files with matching transcripts, and the model's own
untuned voice is offered when there is no reference.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL_REPO = "Audio8/Audio8-TTS-Preview-0.1b"
MODEL_DIRNAME = "Audio8-TTS-Preview-0.1b"

ENGINE_ID = "audio8"

# Measured on this hardware: on CPU the model runs about 10x slower than
# realtime — 43 s of compute for 4 s of speech — while a CUDA device turns that
# around. The old 8000-character ceiling would mean a quarter of an hour of
# silence before a word came out, so the cap follows the device that will
# actually do the work.
MAX_SPEAK_CHARS_CPU = 320
MAX_SPEAK_CHARS_GPU = 2400
# The live value is whatever speak_cap() decides for the current device; this
# is only the floor a caller gets if it asks for a constant.
MAX_SPEAK_CHARS = MAX_SPEAK_CHARS_CPU

# One generate() call covers a sentence or two comfortably; beyond that the
# token budget truncates. Chunk below it and stitch.
MAX_CHUNK_CHARS = 220
MAX_NEW_TOKENS = 512

# Sampling, per the model card's example.
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50

# Generous enough to cover a full CPU synthesis at the cap above, without
# letting a wedged model hold the request forever.
SYNTH_TIMEOUT = 900
MIN_PLAY_TIMEOUT = 20
MAX_PLAY_TIMEOUT = 420
CHARS_PER_SECOND = 15.0

# A beat between stitched chunks, so sentences do not run together.
GAP_SECONDS = 0.18

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

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class Voice(BaseModel):
    id: str
    name: str
    engine: str = ENGINE_ID
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
    # Audio8 exposes no speed or pitch control, so the panel can stop offering
    # sliders that would silently do nothing.
    supports_rate: bool = False
    supports_pitch: bool = False


@dataclass
class _Probe:
    engines: list[EngineInfo] = field(default_factory=list)
    stamp: float = 0.0


_cache = _Probe()
CACHE_SECONDS = 60.0


class DeviceInfo(BaseModel):
    """One place the model could run, and whether it actually can."""

    id: str
    """``cpu``, ``cuda:0``, ``xpu:0``, ``npu`` — what to store in settings."""

    name: str
    available: bool
    detail: str = ""
    install_hint: str = ""


def _cuda_devices() -> list[DeviceInfo]:
    """Every CUDA GPU torch can address, or an explanation of why none."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []

    if torch.cuda.is_available():
        found = []
        for index in range(torch.cuda.device_count()):
            try:
                name = torch.cuda.get_device_name(index)
                total = torch.cuda.get_device_properties(index).total_memory
                detail = f"{total / 1e9:.0f} GB"
            except Exception:  # noqa: BLE001
                name, detail = f"CUDA device {index}", ""
            found.append(
                DeviceInfo(id=f"cuda:{index}", name=name, available=True, detail=detail)
            )
        return found

    # A GPU the driver can see but torch cannot is worth listing anyway: the
    # fix is a different torch build, and silently hiding the card makes that
    # look like missing hardware.
    present = _driver_gpus()
    return [
        DeviceInfo(
            id=f"cuda:{index}",
            name=name,
            available=False,
            detail="present, but this torch build has no CUDA support",
            install_hint=(
                "Install a CUDA build in the gateway venv: .venv/bin/pip install "
                "--index-url https://download.pytorch.org/whl/cu130 torch"
            ),
        )
        for index, name in enumerate(present)
    ]


def _driver_gpus() -> list[str]:
    """NVIDIA GPUs the driver reports, independent of torch."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return []
    try:
        import subprocess

        out = subprocess.run(
            [binary, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _xpu_devices() -> list[DeviceInfo]:
    """Intel GPUs, which torch reaches through XPU rather than CUDA."""
    try:
        import torch

        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return []
        return [
            DeviceInfo(
                id=f"xpu:{i}",
                name=xpu.get_device_name(i),
                available=True,
            )
            for i in range(xpu.device_count())
        ]
    except Exception:  # noqa: BLE001
        return []


def _npu_device() -> DeviceInfo | None:
    """The Intel NPU — listed, and honestly unavailable.

    Keylane's router runs on it, so people will reasonably expect speech to as
    well. It cannot: the NPU is reachable only through OpenVINO, and Audio8 is
    a PyTorch model with custom modelling code and no OpenVINO export. Saying
    that plainly beats leaving the option out and looking broken.
    """
    try:
        import openvino as ov

        if not any(d == "NPU" or d.startswith("NPU.") for d in ov.Core().available_devices):
            return None
    except Exception:  # noqa: BLE001
        return None
    return DeviceInfo(
        id="npu",
        name="Intel NPU",
        available=False,
        detail=(
            "reachable only through OpenVINO, and Audio8 is a PyTorch model "
            "with no OpenVINO export — the NPU cannot run it"
        ),
    )


def available_devices() -> list[DeviceInfo]:
    """Everywhere speech could run, best first, each with its own verdict."""
    devices: list[DeviceInfo] = []
    devices.extend(_cuda_devices())
    devices.extend(_xpu_devices())
    devices.append(
        DeviceInfo(
            id="cpu",
            name="CPU",
            available=True,
            detail="always works, roughly 10x slower than realtime",
        )
    )
    npu = _npu_device()
    if npu is not None:
        devices.append(npu)
    return devices


def resolve_device(preference: str = "") -> str:
    """The torch device to use, honouring ``preference`` when it can.

    Falls back rather than failing: a saved preference for a GPU that is not
    there any more should still produce speech, on the CPU, rather than an
    error about hardware.
    """
    devices = {d.id: d for d in available_devices()}
    chosen = devices.get(preference)
    if chosen is not None and chosen.available:
        return chosen.id
    if preference and preference not in {"", "auto"}:
        logger.info(
            "Speech device %r is not usable (%s); falling back.",
            preference,
            chosen.detail if chosen else "no such device",
        )
    # auto: the fastest thing that actually works.
    for device in devices.values():
        if device.available and device.id != "cpu":
            return device.id
    return "cpu"


def torch_device(preference: str = "") -> str:
    """Backwards-compatible alias; ``cuda``/``cpu`` for callers that branch."""
    return resolve_device(preference)


def is_accelerated(device: str) -> bool:
    """Whether a resolved device is something other than the CPU."""
    return bool(device) and not device.startswith("cpu")


def speak_cap(device: str = "") -> int:
    """How much text is worth attempting on the device that will run it."""
    resolved = device or resolve_device(_configured_device())
    return MAX_SPEAK_CHARS_GPU if is_accelerated(resolved) else MAX_SPEAK_CHARS_CPU


def _configured_device() -> str:
    """The device the user picked under Assistant -> Read aloud."""
    try:
        from app.assistant_settings import load_assistant_settings

        return load_assistant_settings().speech.device or "auto"
    except Exception:  # noqa: BLE001
        return "auto"


def clean_for_speech(text: str, limit: int | None = None) -> str:
    """Strip markup so the synthesiser reads words, not punctuation."""
    if limit is None:
        limit = speak_cap()
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


def split_for_synthesis(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Break text into pieces one ``generate`` call can finish.

    Splitting on sentences keeps the prosody of each piece intact; a sentence
    longer than the limit is broken on whitespace as a last resort rather than
    mid-word.
    """
    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text.strip()):
        if not sentence:
            continue
        if len(sentence) > limit:
            if current:
                chunks.append(current)
                current = ""
            words, line = sentence.split(), ""
            for word in words:
                if len(line) + len(word) + 1 > limit:
                    if line:
                        chunks.append(line)
                    line = word
                else:
                    line = f"{line} {word}".strip()
            if line:
                current = line
            continue
        if len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks or ([text.strip()] if text.strip() else [])


# ------------------------------------------------------------------ model --


def model_dir() -> Path:
    from app.config import get_config

    return get_config().resolve_path(Path("models") / "tts" / MODEL_DIRNAME)


def voices_dir() -> Path:
    from app.config import get_config

    return get_config().resolve_path(Path("voices"))


def model_installed() -> bool:
    """Whether the weights are actually on disk, not merely a directory."""
    directory = model_dir()
    weights = directory / "model.safetensors"
    codec = directory / "codec.pth"
    try:
        return (
            weights.is_file()
            and weights.stat().st_size > 1_000_000
            and codec.is_file()
            and codec.stat().st_size > 1_000_000
        )
    except OSError:
        return False


def _missing_dependency() -> str:
    """The first import the engine needs and cannot find, or ``''``."""
    for module, package in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("soundfile", "soundfile"),
    ):
        try:
            __import__(module)
        except Exception:  # noqa: BLE001
            return package
    return ""


class _Synth:
    """The loaded model, built once and reused.

    Loading reads ~1.7 GB off disk and takes seconds, so it happens on first
    use rather than at import, and behind a lock so two callers cannot both
    pay for it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processor: Any = None
        self._model: Any = None
        self._sample_rate = 0
        self._device = "cpu"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def device(self) -> str:
        return self._device

    def load(self, device: str | None = None) -> None:
        target = device or resolve_device(_configured_device())
        with self._lock:
            # Switching device means rebuilding: the weights live on the old
            # one, and moving a model mid-flight is not worth the subtlety.
            if self._model is not None and self._device == target:
                return
            if self._model is not None:
                logger.info("Speech device changed %s -> %s; reloading", self._device, target)
                self._model = None
                self._processor = None
            import torch
            from transformers import AutoModel, AutoProcessor

            path = str(model_dir())
            device = target
            # bfloat16 is a win on a GPU and a liability on a CPU, where it is
            # emulated rather than native.
            dtype = torch.bfloat16 if is_accelerated(device) else torch.float32
            logger.info("Loading Audio8 TTS from %s on %s", path, device)
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            model = (
                AutoModel.from_pretrained(path, trust_remote_code=True, dtype=dtype)
                .eval()
                .to(device)
            )
            self._processor = processor
            self._model = model
            self._device = device
            self._sample_rate = int(getattr(model.config, "codec_sample_rate", 44100))
            logger.info("Audio8 TTS ready on %s at %d Hz", device, self._sample_rate)

    def synthesize(
        self, text: str, reference: tuple[str, str] | None, device: str | None = None
    ) -> Any:
        """Return a float32 mono numpy array for one chunk of text."""
        import torch

        self.load(device)
        kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if reference is not None:
            audio_path, transcript = reference
            kwargs["reference_audio"] = [audio_path]
            kwargs["reference_text"] = [transcript]

        inputs = self._processor(**kwargs)
        inputs = {
            name: value.to(self._device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                do_sample=True,
                return_dict_in_generate=True,
            )
            waveforms, lengths = self._model.decode_audio(output.codes)
        return waveforms[0, : int(lengths[0])].float().cpu().numpy()


_synth = _Synth()


def _render_wav(
    text: str, reference: tuple[str, str] | None, device: str | None = None
) -> bytes:
    """Synthesise every chunk and return one WAV, ready to play."""
    import numpy as np
    import soundfile as sf

    pieces = []
    for chunk in split_for_synthesis(text):
        pieces.append(_synth.synthesize(chunk, reference, device))

    rate = _synth.sample_rate or 44100
    if len(pieces) > 1:
        gap = np.zeros(int(rate * GAP_SECONDS), dtype=np.float32)
        stitched = np.concatenate(
            [part for piece in pieces for part in (piece, gap)][:-1]
        )
    else:
        stitched = pieces[0]

    buffer = io.BytesIO()
    sf.write(buffer, stitched, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# ----------------------------------------------------------------- voices --


def _reference_voices() -> list[Voice]:
    """Reference clips to clone, from the voices directory.

    A voice is ``<name>.wav`` beside ``<name>.txt`` holding exactly what the
    clip says — the model needs the transcript to condition on. A clip without
    one is skipped rather than guessed at, since a wrong transcript degrades
    the clone badly.
    """
    directory = voices_dir()
    if not directory.is_dir():
        return []
    voices: list[Voice] = []
    for clip in sorted(directory.glob("*.wav")):
        transcript = clip.with_suffix(".txt")
        if not transcript.is_file():
            logger.debug("Skipping %s: no matching .txt transcript", clip.name)
            continue
        voices.append(
            Voice(
                id=str(clip),
                name=clip.stem.replace("_", " ").replace("-", " ").title(),
                quality="cloned",
            )
        )
    return voices


def _voices() -> list[Voice]:
    # The model's own voice needs no reference, so it is always offered first.
    return [
        Voice(id="", name="Audio8 default", quality="built-in"),
        *_reference_voices(),
    ]


def _reference_for(voice_id: str) -> tuple[str, str] | None:
    if not voice_id:
        return None
    clip = Path(voice_id)
    transcript = clip.with_suffix(".txt")
    if not clip.is_file() or not transcript.is_file():
        return None
    try:
        text = transcript.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return (str(clip), text) if text else None


# -------------------------------------------------------------- discovery --


def probe_engines(*, refresh: bool = False) -> list[EngineInfo]:
    """What this machine can actually speak with."""
    import time

    now = time.monotonic()
    if _cache.engines and not refresh and (now - _cache.stamp) < CACHE_SECONDS:
        return _cache.engines

    missing = _missing_dependency()
    installed = model_installed()
    available = installed and not missing

    if missing:
        detail = f"the {missing} package is not installed"
        hint = f"pip install {missing}"
    elif not installed:
        detail = "the model has not been downloaded yet"
        hint = (
            f"Download {MODEL_REPO} into {model_dir()} — "
            "about 1.7 GB. The Models tab can fetch it."
        )
    else:
        count = len(_reference_voices())
        device = resolve_device(_configured_device())
        named = next(
            (d.name for d in available_devices() if d.id == device), device
        )
        voices_note = (
            f"{count} cloned voice{'s' if count != 1 else ''}"
            if count
            else f"add a wav and matching txt to {voices_dir()} to clone a voice"
        )
        # Speed is the difference between a usable feature and a novelty here,
        # and it is entirely down to the device, so say which one it found.
        speed = (
            f"running on {named}"
            if is_accelerated(device)
            else (
                f"running on {named}, roughly 10x slower than realtime — long "
                "answers are truncated"
            )
        )
        detail = f"ready, {speed}. {voices_note}."
        # If a faster device exists but cannot be used, its own hint is more
        # specific than anything generic said here.
        blocked = next(
            (d for d in available_devices() if not d.available and d.install_hint), None
        )
        hint = "" if is_accelerated(device) else (blocked.install_hint if blocked else "")

    engines = [
        EngineInfo(
            id=ENGINE_ID,
            name="Audio8 TTS",
            available=available,
            description=(
                "Neural speech that runs on this machine. Zero-shot: point it at "
                "a short recording and it speaks in that voice."
            ),
            detail=detail,
            install_hint=hint,
            voices=_voices() if available else [],
        )
    ]

    _cache.engines = engines
    _cache.stamp = now
    return engines


def default_engine() -> str:
    for engine in probe_engines():
        if engine.available:
            return engine.id
    return ""


def resolve(engine_id: str, voice_id: str) -> tuple[EngineInfo | None, str]:
    """Pick the engine and voice to use, falling back sensibly."""
    engines = {e.id: e for e in probe_engines()}
    engine = engines.get(engine_id)
    if engine is None or not engine.available:
        # A settings file written when piper or espeak was the engine must not
        # strand the user on an id that no longer exists. The voice survives
        # this: a caller that asked for one explicitly still means it, and the
        # check below drops it anyway if it belongs to the retired engine.
        engine = engines.get(default_engine())
    if engine is None:
        return None, ""
    if voice_id and any(v.id == voice_id for v in engine.voices):
        return engine, voice_id
    return engine, ""


# ------------------------------------------------------------- synthesis --


class SpeakError(RuntimeError):
    pass


def _player() -> list[str] | None:
    """How to play a WAV handed over on stdin."""
    for binary, args in (
        ("pw-play", ["-"]),
        ("paplay", ["-"]),
        ("aplay", ["-q", "-"]),
    ):
        path = shutil.which(binary)
        if path:
            return [path, *args]
    return None


# The player currently speaking, so stop() can silence it. Synthesis is now
# in-process, so there is no synthesiser process to signal — only playback.
_playing: asyncio.subprocess.Process | None = None
_playing_lock = asyncio.Lock()

# Synthesis is the long part — minutes on a CPU — and it runs on a thread that
# cannot be interrupted. Counting stop requests lets a finished synthesis
# notice it was cancelled while it worked, and decline to start speaking.
_stop_epoch = 0


async def speak(
    text: str,
    *,
    engine_id: str = "",
    voice_id: str = "",
    rate: int = 100,
    pitch: int = 50,
    device: str = "",
) -> dict[str, Any]:
    """Read text aloud. Returns what was actually used.

    ``rate`` and ``pitch`` are accepted so existing callers and stored settings
    keep working, but Audio8 exposes no speed or pitch control and they are
    ignored. The engine advertises that via ``supports_rate``/``supports_pitch``.
    """
    target = resolve_device(device or _configured_device())
    body = clean_for_speech(text, speak_cap(target))
    if not body:
        raise SpeakError("There is nothing to read.")

    engine, voice = resolve(engine_id, voice_id)
    if engine is None or not engine.available:
        missing = _missing_dependency()
        if missing:
            raise SpeakError(f"Read aloud needs the {missing} package: pip install {missing}")
        raise SpeakError(
            f"The speech model is not installed. Download {MODEL_REPO} "
            f"into {model_dir()}."
        )

    player = _player()
    if player is None:
        raise SpeakError("No audio player found. Try: sudo dnf install pipewire-utils")

    reference = _reference_for(voice)
    if voice and reference is None:
        raise SpeakError(
            f"The voice clip {voice} has no readable transcript beside it. "
            "Each voice needs a .txt file with exactly what the clip says."
        )

    logger.info("Speaking with Audio8 (%s)", voice or "default voice")
    epoch = _stop_epoch
    try:
        # Torch inference is blocking and slow; keeping it off the event loop
        # is what lets the gateway answer anything else while it speaks.
        wav = await asyncio.wait_for(
            asyncio.to_thread(_render_wav, body, reference, target),
            timeout=SYNTH_TIMEOUT,
        )
    except asyncio.TimeoutError as exc:
        raise SpeakError("Speech synthesis timed out.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Audio8 synthesis failed")
        raise SpeakError(f"Audio8 could not synthesise that: {exc}"[:300]) from exc

    # Someone pressed stop while this was being generated. Starting playback
    # now would speak something they already dismissed.
    if _stop_epoch != epoch:
        logger.info("Discarding synthesis: stop was requested while generating")
        return {
            "engine": engine.id,
            "voice": voice,
            "characters": len(body),
            "rate": rate,
            "sample_rate": _synth.sample_rate,
            "device": _synth.device,
            "stopped": True,
        }

    global _playing
    try:
        process = await asyncio.create_subprocess_exec(
            *player,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise SpeakError(f"{exc.filename} is not installed.") from exc

    async with _playing_lock:
        _playing = process

    budget = min(
        max(len(body) / CHARS_PER_SECOND + 10, MIN_PLAY_TIMEOUT), MAX_PLAY_TIMEOUT
    )
    try:
        _out, err = await asyncio.wait_for(process.communicate(wav), timeout=budget)
    except asyncio.TimeoutError as exc:
        if process.returncode is None:
            process.kill()
        raise SpeakError("Playback timed out.") from exc
    finally:
        async with _playing_lock:
            if _playing is process:
                _playing = None

    # A killed player is a deliberate stop(), not a failure worth raising over.
    if process.returncode not in (0, None, -9, -15):
        detail = err.decode("utf-8", "replace").strip().splitlines()
        raise SpeakError(
            f"Playback failed: {detail[-1][:200] if detail else process.returncode}"
        )

    return {
        "engine": engine.id,
        "voice": voice,
        "characters": len(body),
        "rate": rate,
        "sample_rate": _synth.sample_rate,
        "device": _synth.device,
    }


async def stop() -> int:
    """Silence anything speaking, and cancel a synthesis still in flight.

    Returns how many players were killed. A synthesis that has not finished
    cannot be interrupted, but it will throw its audio away rather than start
    speaking — so this reads as "stop" either way.
    """
    global _playing, _stop_epoch
    _stop_epoch += 1
    async with _playing_lock:
        process = _playing
        _playing = None
    if process is None or process.returncode is not None:
        return 0
    try:
        process.kill()
    except ProcessLookupError:
        return 0
    return 1
