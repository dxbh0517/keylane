"""Thin HTTP client for the Keylane gateway, used by the popup and the tray.

Every call is synchronous and short-lived; callers run them on worker threads
and marshal results back with ``GLib.idle_add``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Iterator

import httpx

logger = logging.getLogger(__name__)

DEFAULT_GATEWAY = os.environ.get("KEYLANE_GATEWAY", "http://127.0.0.1:9100")


class GatewayClient:
    def __init__(self, base_url: str = DEFAULT_GATEWAY) -> None:
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------ reads

    def _get(self, path: str, *, timeout: float = 3.0) -> Any:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{self.base_url}{path}")
            response.raise_for_status()
            return response.json()

    def status(self) -> dict[str, Any]:
        try:
            return self._get("/api/status")
        except Exception:  # noqa: BLE001
            return {"gateway": False, "npu": False}

    def activity(self) -> dict[str, Any]:
        try:
            return self._get("/api/activity")
        except Exception:  # noqa: BLE001
            return {"busy": False, "active_count": 0, "needs_attention": 0, "active": []}

    def projects(self) -> list[dict[str, str]]:
        try:
            return self._get("/api/projects").get("projects", [])
        except Exception:  # noqa: BLE001
            return []

    def gateway_config(self) -> dict[str, Any]:
        try:
            return self._get("/api/config")
        except Exception:  # noqa: BLE001
            return {}

    def active_theme(self) -> dict[str, Any] | None:
        try:
            return self._get("/api/themes/active")
        except Exception:  # noqa: BLE001
            return None

    def launcher_css(self) -> str:
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.base_url}/api/themes/active/launcher.css")
                if response.status_code == 200:
                    return response.text
        except Exception:  # noqa: BLE001
            pass
        return ""

    # ----------------------------------------------------------------- writes

    def speech_available(self) -> bool:
        """Whether read-aloud is switched on and an engine is installed."""
        try:
            data = self._get("/api/speech")
        except Exception:  # noqa: BLE001
            return False
        return bool(data.get("available")) and bool(
            (data.get("settings") or {}).get("enabled")
        )

    def speak(self, text: str) -> tuple[bool, str]:
        """Read text aloud. Returns ``(ok, detail)``; blocks until finished."""
        try:
            with httpx.Client(timeout=600.0) as client:
                response = client.post(
                    f"{self.base_url}/api/speech/speak", json={"text": text}
                )
                if response.status_code >= 400:
                    payload = response.json() if response.content else {}
                    return False, str(payload.get("detail") or response.text)[:200]
                return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:200]

    def stop_speech(self) -> None:
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(f"{self.base_url}/api/speech/stop")
        except Exception:  # noqa: BLE001
            pass

    def chat(self, payload: dict[str, Any], *, timeout: float = 900.0) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(f"{self.base_url}/api/chat", json=payload)
                data = response.json() if response.content else {}
                if response.status_code >= 400 and "error" not in data:
                    data = {
                        "status": "failed",
                        "error": data.get("detail") or response.text,
                        "task_id": "",
                    }
                return data
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc), "task_id": ""}

    def cancel(self, task_id: str) -> None:
        try:
            with httpx.Client(timeout=5.0) as client:
                client.post(f"{self.base_url}/api/tasks/{task_id}/cancel")
        except Exception:  # noqa: BLE001
            pass

    def transcribe(self, wav_bytes: bytes) -> tuple[str, str]:
        """Returns ``(text, error)``."""
        try:
            with httpx.Client(timeout=180.0) as client:
                response = client.post(
                    f"{self.base_url}/api/transcribe",
                    files={"file": ("speech.wav", wav_bytes, "audio/wav")},
                )
                payload = response.json() if response.content else {}
                if response.status_code >= 400:
                    detail = payload.get("detail") if isinstance(payload, dict) else None
                    return "", str(detail or response.text or "Transcription failed")
                text = str(payload.get("text") or "").strip()
                return (text, "") if text else ("", "No speech detected")
        except Exception as exc:  # noqa: BLE001
            return "", str(exc)

    # -------------------------------------------------------------- streaming

    def stream_events(
        self,
        on_snapshot: Callable[[dict[str, Any]], None],
        should_stop: Callable[[], bool],
    ) -> None:
        """Consume /api/events until ``should_stop()`` or the stream drops.

        Raises on connection failure so the caller can back off and retry.
        """
        with httpx.Client(timeout=httpx.Timeout(10.0, read=None)) as client:
            with client.stream("GET", f"{self.base_url}/api/events") as response:
                response.raise_for_status()
                event_name = ""
                for line in response.iter_lines():
                    if should_stop():
                        return
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        payload = line[5:].strip()
                        if event_name != "snapshot":
                            continue
                        try:
                            on_snapshot(json.loads(payload))
                        except json.JSONDecodeError:
                            logger.debug("Bad SSE payload: %s", payload[:120])


def iter_sse(response: httpx.Response) -> Iterator[tuple[str, str]]:
    """Yield ``(event, data)`` pairs from an SSE response."""
    event_name = "message"
    for line in response.iter_lines():
        if not line:
            event_name = "message"
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            yield event_name, line[5:].strip()
