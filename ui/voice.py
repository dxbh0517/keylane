"""Voice input via Whisper."""

from __future__ import annotations

import logging
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def record_and_transcribe(
    *,
    seconds: float = 5.0,
    on_done: Callable[[str], None],
    on_error: Callable[[str], None],
) -> None:
    """Record from default mic and transcribe in a background thread."""

    def _work() -> None:
        try:
            import numpy as np
            import sounddevice as sd
            import whisper

            rate = 16000
            frames = int(seconds * rate)
            audio = sd.rec(frames, samplerate=rate, channels=1, dtype="float32")
            sd.wait()

            path = Path(tempfile.mkdtemp()) / "clip.wav"
            pcm = (audio.flatten() * 32767).astype(np.int16)
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(pcm.tobytes())

            model = whisper.load_model("base")
            result = model.transcribe(str(path), fp16=False)
            text = str(result.get("text", "")).strip()
            on_done(text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("voice input failed")
            on_error(str(exc))

    threading.Thread(target=_work, daemon=True).start()
