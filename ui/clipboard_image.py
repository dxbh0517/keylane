"""Read images from the GTK clipboard."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import gi

gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib  # type: ignore[attr-defined]


def _texture_to_png(texture: Gdk.Texture) -> bytes | None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        texture.save_to_png(str(path))
        return path.read_bytes()
    except Exception:  # noqa: BLE001
        return None
    finally:
        path.unlink(missing_ok=True)


def read_image_bytes(on_ready: Callable[[bytes | None], None]) -> None:
    """Async clipboard read; invokes *on_ready* with PNG bytes or None."""

    def _finish(_provider: Gdk.ContentProvider, result) -> None:
        try:
            value = _provider.read_value_finish(result)
        except Exception:  # noqa: BLE001
            GLib.idle_add(on_ready, None)
            return

        data: bytes | None = None
        if isinstance(value, Gdk.Texture):
            data = _texture_to_png(value)

        GLib.idle_add(on_ready, data)

    display = Gdk.Display.get_default()
    if not display:
        on_ready(None)
        return
    clipboard = display.get_clipboard()
    clipboard.read_texture_async(None, _finish)
