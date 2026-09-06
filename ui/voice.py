"""Voice input via Whisper.

Two consumers now: the mic button in the Spotlight bar, which fills the entry
box, and :mod:`ui.dictation`, which types into somebody else's window. Both go
through :func:`transcribe_file`, and both want the same thing from it —
**the model stays loaded**.

That is the change worth spelling out. Whisper used to be loaded inside the
worker thread on every utterance, which is a disk read and a few seconds of
setup paid per sentence. For a mic button pressed occasionally that was merely
wasteful; for dictation it is the whole latency budget. The model is cached
against its name, so switching size in Settings still takes effect.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Whisper's own names, smallest first. `base` is the default because it loads
# in about a second; `small` is the first size that reliably keeps technical
# vocabulary intact, which is what dictation is usually full of.
MODEL_SIZES = ("tiny", "base", "small", "medium", "large")

_model_lock = threading.Lock()
_model: Any = None
_model_name = ""


def load_model(name: str = "base") -> Any:
    """The Whisper model, loaded once and kept."""
    global _model, _model_name
    with _model_lock:
        if _model is None or _model_name != name:
            import whisper

            logger.info("loading whisper model %r", name)
            _model = whisper.load_model(name)
            _model_name = name
        return _model


def loaded_model() -> str:
    """Which model is resident, or "" if none has been loaded yet."""
    return _model_name


def write_wav(samples: Any, path: Path) -> Path:
    """Write float32 mono samples as the 16 kHz PCM WAV Whisper expects."""
    import numpy as np

    pcm = (samples.flatten() * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return path


def transcribe_file(
    path: Path,
    *,
    model: str = "base",
    language: str = "",
    initial_prompt: str = "",
) -> str:
    """Transcribe a WAV file.

    ``initial_prompt`` is how a custom vocabulary actually reaches Whisper:
    it conditions the decoder, so listing the proper nouns and jargon a user
    dictates is what stops "Keylane" coming back as "key lane". A post-hoc
    find-and-replace cannot do the same job, because by then the surrounding
    words have already been decoded around the wrong one.
    """
    options: dict[str, Any] = {"fp16": False}
    if language:
        options["language"] = language
    if initial_prompt:
        options["initial_prompt"] = initial_prompt
    result = load_model(model).transcribe(str(path), **options)
    return str(result.get("text", "")).strip()


class MicRecorder:
    """Toggle mic on/off; hand the captured audio to a callback when stopped."""

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

    def _take(self) -> tuple[Any, list]:
        with self._lock:
            stream = self._stream
            chunks = list(self._chunks)
            self._stream = None
            self._chunks = []
            self._recording = False
        return stream, chunks

    def stop_to_wav(self) -> Path | None:
        """Stop recording and write what was captured. Runs on the caller's thread."""
        stream, chunks = self._take()
        if stream is not None:
            stream.stop()
            stream.close()
        if not chunks:
            return None
        import numpy as np

        audio = np.concatenate(chunks, axis=0)
        return write_wav(audio, Path(tempfile.mkdtemp()) / "clip.wav")

    def stop(
        self,
        *,
        on_done: Callable[[str], None],
        on_error: Callable[[str], None],
        model: str = "base",
    ) -> None:
        """Stop and transcribe on a worker thread."""
        if not self.recording:
            return

        def _work() -> None:
            try:
                path = self.stop_to_wav()
                if path is None:
                    on_done("")
                    return
                on_done(transcribe_file(path, model=model))
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
    model: str = "base",
) -> None:
    _recorder.stop(on_done=on_done, on_error=on_error, model=model)


def toggle_mic(
    *,
    on_done: Callable[[str], None],
    on_error: Callable[[str], None],
    model: str = "base",
) -> bool:
    """Start or stop recording. Returns True if now recording."""
    if _recorder.recording:
        _recorder.stop(on_done=on_done, on_error=on_error, model=model)
        return False
    _recorder.start()
    return True
