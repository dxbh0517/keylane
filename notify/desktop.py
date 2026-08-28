"""Desktop notifications via notify-send."""

from __future__ import annotations

import shutil
import subprocess


def send_notification(title: str, body: str) -> bool:
    if not shutil.which("notify-send"):
        return False
    subprocess.run(
        ["notify-send", title, body[:500]],
        check=False,
    )
    return True
