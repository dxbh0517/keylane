"""Push-to-talk transcription (CPU Whisper first; NPU later)."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        import whisper  # type: ignore

        # tiny is fast enough for push-to-talk on CPU; upgrade via KEYLANE_WHISPER_MODEL
        import os

        model_name = os.environ.get("KEYLANE_WHISPER_MODEL", "tiny")
        _whisper_model = whisper.load_model(model_name)
        return _whisper_model
    except Exception as exc:  # noqa: BLE001
        logger.warning("openai-whisper unavailable: %s", exc)
        return None


async def transcribe_wav_bytes(data: bytes, language: str | None = "en") -> str:
    """Transcribe audio bytes. Prefers whisper; falls back to empty with error."""
    if not data:
        raise ValueError("Empty audio payload")

    model = _get_whisper()
    if model is None:
        raise RuntimeError(
            "Transcription unavailable. Install openai-whisper in the gateway "
            "venv (`pip install openai-whisper`) and ensure ffmpeg is installed."
        )

    suffix = ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = Path(tmp.name)

    try:
        result = model.transcribe(str(path), language=language or "en")
        return (result.get("text") or "").strip()
    finally:
        path.unlink(missing_ok=True)


def wav_from_pcm16(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw PCM16 LE into a WAV container."""
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buffer.getvalue()
