#!/usr/bin/env bash
# Resolve Python for GTK UI (needs system PyGObject on Fedora).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${KEYLANE_VENV:-$ROOT/.venv}"
VENV_PY="${VENV}/bin/python"

if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import gi" >/dev/null 2>&1; then
  echo "$VENV_PY"
  exit 0
fi

if command -v python3 >/dev/null 2>&1 && python3 -c "import gi" >/dev/null 2>&1; then
  echo "python3"
  exit 0
fi

echo "Keylane UI needs PyGObject (python3-gobject). Install with: sudo dnf install python3-gobject gtk4" >&2
exit 1
