#!/usr/bin/env python3
"""Keylane GTK4/libadwaita launcher entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo without installing as a package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from launcher.window import run_launcher


def main() -> int:
    return run_launcher()


if __name__ == "__main__":
    raise SystemExit(main())
