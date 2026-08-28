"""Optional TTS for notifications and spoken answers."""

from __future__ import annotations

import logging
import threading

from daemon.config import assistant_settings

logger = logging.getLogger(__name__)


def maybe_speak_notify(text: str) -> None:
    notify_cfg = assistant_settings().get("notify", {})
    if not notify_cfg.get("tts_on_notify", False):
        return
    speak_text(text)


def speak_text(text: str) -> None:
    def _run() -> None:
        try:
            from tts.engine import speak

            speak(text)
        except Exception:  # noqa: BLE001
            logger.debug("TTS failed", exc_info=True)

    threading.Thread(target=_run, daemon=True).start()
