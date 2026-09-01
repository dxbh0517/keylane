"""Voice input via Whisper — push-to-talk / toggle recording."""

from __future__ import annotations

import logging
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class MicRecorder:
    """Toggle mic on/off; transcribe when recording stops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stream = None
        self._chunks: list = []
        self._recording = False

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            import sounddevice as sd

            self._chunks = []

            def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
                self._chunks.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=_callback,
            )
            self._stream.start()
            self._recording = True

    def stop(
        self,
        *,
        on_done: Callable[[str], None],
        on_error: Callable[[str], None],
    ) -> None:
        with self._lock:
            if not self._recording:
                return
            stream = self._stream
            chunks = list(self._chunks)
            self._stream = None
            self._chunks = []
            self._recording = False

        def _work() -> None:
            try:
                if stream is not None:
                    stream.stop()
                    stream.close()
                if not chunks:
                    on_done("")
                    return

                import numpy as np
                import whisper

                audio = np.concatenate(chunks, axis=0)
                path = Path(tempfile.mkdtemp()) / "clip.wav"
                pcm = (audio.flatten() * 32767).astype(np.int16)
                with wave.open(str(path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(pcm.tobytes())

                model = whisper.load_model("base")
                result = model.transcribe(str(path), fp16=False)
                on_done(str(result.get("text", "")).strip())
            except Exception as exc:  # noqa: BLE001
                logger.exception("voice input failed")
                on_error(str(exc))

        threading.Thread(target=_work, daemon=True).start()


_recorder = MicRecorder()


def mic_recording() -> bool:
    return _recorder.recording


def start_mic() -> None:
    _recorder.start()


def stop_mic(
    *,
    on_done: Callable[[str], None],
    on_error: Callable[[str], None],
) -> None:
    _recorder.stop(on_done=on_done, on_error=on_error)


def toggle_mic(
    *,
    on_done: Callable[[str], None],
    on_error: Callable[[str], None],
) -> bool:
    """Start or stop recording. Returns True if now recording."""
    if _recorder.recording:
        _recorder.stop(on_done=on_done, on_error=on_error)
        return False
    _recorder.start()
    return True


def record_and_transcribe(
    *,
    seconds: float = 5.0,
    on_done: Callable[[str], None],
    on_error: Callable[[str], None],
) -> None:
    """Fixed-length capture (legacy helper)."""

    def _work() -> None:
        try:
            import numpy as np
            import sounddevice as sd
            import whisper

            frames = int(seconds * SAMPLE_RATE)
            audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()

            path = Path(tempfile.mkdtemp()) / "clip.wav"
            pcm = (audio.flatten() * 32767).astype(np.int16)
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(pcm.tobytes())

            model = whisper.load_model("base")
            result = model.transcribe(str(path), fp16=False)
            on_done(str(result.get("text", "")).strip())
        except Exception as exc:  # noqa: BLE001
            logger.exception("voice input failed")
            on_error(str(exc))

    threading.Thread(target=_work, daemon=True).start()
