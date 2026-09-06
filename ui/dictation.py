"""Hold a key, talk, and have the words land in the app you were already in.

Two modes share one pipeline, because they differ only in what turns audio
into the string that gets typed:

* **dictate** — Whisper transcribes what you said, and that is the text.
* **compose** — Whisper transcribes what you *want*, Keylane looks at the
  screen, and the model writes the text for you. You dictate the intent; it
  drafts the reply.

The target window is captured when recording **starts**, not when it ends.
That ordering is the whole reliability story: by the time a transcript exists
the user may have alt-tabbed, and the compositor's idea of "focused" would send
their sentence into the wrong window.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from daemon.paths import DATA
from inject import InjectionError, InjectionTarget
from inject import type_text as inject_text
from ui.voice import MicRecorder, transcribe_file
from ui.window_ctx import UNKNOWN, FocusedWindow, focused_window

logger = logging.getLogger(__name__)

TERMS_PATH = DATA / "dictation_terms.txt"

# Whisper's `initial_prompt` is a conditioning context, not a glossary, so it
# has to read like a sentence for the decoder to make use of it.
_TERMS_PREFIX = "The speaker often uses these words: "
# Long enough for a real vocabulary, short enough not to crowd out the audio.
TERMS_MAX_CHARS = 800


@dataclass
class DictationSettings:
    """The `[dictation]` section, resolved once per utterance."""

    enabled: bool = True
    model: str = "base"
    language: str = ""
    cleanup: bool = True
    provider: str = "auto"

    @classmethod
    def load(cls) -> DictationSettings:
        try:
            from daemon.config import get_section

            raw = get_section("dictation")
        except Exception:  # noqa: BLE001
            logger.debug("could not read dictation settings; using defaults", exc_info=True)
            raw = {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            model=str(raw.get("model", "base") or "base"),
            language=str(raw.get("language", "") or ""),
            cleanup=bool(raw.get("cleanup", True)),
            provider=str(raw.get("provider", "auto") or "auto"),
        )


@dataclass
class DictationResult:
    """What happened, in enough detail for the HUD to say something useful."""

    text: str = ""
    provider: str = ""
    target: FocusedWindow = field(default_factory=lambda: UNKNOWN)
    injected: bool = False
    error: str = ""


def load_terms() -> str:
    """The user's custom vocabulary, as a sentence Whisper can be primed with."""
    try:
        raw = TERMS_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""
    words = [line.strip() for line in raw.splitlines()]
    words = [w for w in words if w and not w.startswith("#")]
    if not words:
        return ""
    sentence = _TERMS_PREFIX + ", ".join(words) + "."
    return sentence[:TERMS_MAX_CHARS]


def _cleanup_via_daemon(text: str) -> str:
    """Punctuation and casing only, on the utility route.

    A failure here is not a failure of dictation: the raw transcript is still
    the user's words, and losing it to a daemon that is not running would be a
    far worse outcome than leaving it uncapitalised.
    """
    try:
        from ui import api

        response = api.post("/dictation/cleanup", json={"text": text}, timeout=30)
        response.raise_for_status()
        cleaned = str(response.json().get("text", "")).strip()
    except Exception as exc:  # noqa: BLE001
        logger.info("dictation cleanup unavailable (%s); using the raw transcript", exc)
        return text
    return cleaned or text


def _compose_via_daemon(instruction: str, image: bytes | None, target: FocusedWindow) -> str:
    """Ask the model to write the text, given the screen and the instruction."""
    import base64

    from ui import api

    payload = {
        "instruction": instruction,
        "app": target.label(),
        "image": base64.b64encode(image).decode("ascii") if image else "",
    }
    response = api.post("/compose", json=payload, timeout=300)
    response.raise_for_status()
    return str(response.json().get("text", "")).strip()


class Dictation:
    """One recorder, driven by whichever hotkey started it."""

    def __init__(self) -> None:
        self._recorder = MicRecorder()
        self._lock = threading.Lock()
        self._target: FocusedWindow = UNKNOWN
        self._mode = "dictate"

    @property
    def active(self) -> bool:
        return self._recorder.recording

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def target(self) -> FocusedWindow:
        return self._target

    def start(self, mode: str = "dictate") -> FocusedWindow:
        """Begin recording, remembering where the text is going."""
        with self._lock:
            window = focused_window()
            # Keylane's own HUD is never the destination. Where the compositor
            # focused it, we have no better answer than "unknown", and the
            # injection still goes wherever the caret actually is.
            self._target = UNKNOWN if window.is_keylane else window
            self._mode = mode
        self._recorder.start()
        return self._target

    def cancel(self) -> None:
        """Drop the recording without transcribing or typing anything."""
        if self._recorder.recording:
            self._recorder.stop_to_wav()

    def stop(
        self,
        *,
        on_done: Callable[[DictationResult], None],
        capture_screen: Callable[[], bytes | None] | None = None,
    ) -> None:
        """Stop, transcribe, and inject — all on a worker thread."""
        if not self._recorder.recording:
            return
        settings = DictationSettings.load()
        target = self._target
        mode = self._mode

        def _work() -> None:
            result = DictationResult(target=target)
            try:
                wav: Path | None = self._recorder.stop_to_wav()
                if wav is None:
                    on_done(result)
                    return

                spoken = transcribe_file(
                    wav,
                    model=settings.model,
                    language=settings.language,
                    initial_prompt=load_terms(),
                )
                if not spoken:
                    on_done(result)
                    return

                if mode == "compose":
                    image = capture_screen() if capture_screen else None
                    result.text = _compose_via_daemon(spoken, image, target)
                elif settings.cleanup:
                    result.text = _cleanup_via_daemon(spoken)
                else:
                    result.text = spoken

                if not result.text:
                    result.error = "nothing to type"
                    on_done(result)
                    return

                result.provider = inject_text(
                    result.text,
                    InjectionTarget(app_id=target.app_id, title=target.title),
                    prefer=settings.provider,
                )
                result.injected = bool(result.provider)
            except InjectionError as exc:
                # The words are not lost — they are in `result.text`, and the
                # caller shows them so the user can copy them by hand.
                result.error = str(exc)
            except Exception as exc:  # noqa: BLE001
                logger.exception("dictation failed")
                result.error = str(exc)
            on_done(result)

        threading.Thread(target=_work, daemon=True).start()
