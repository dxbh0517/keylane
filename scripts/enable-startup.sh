#!/usr/bin/env bash
# Install and enable Keylane user services (start on login).
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${KEYLANE_DEST:-${HOME}/.local/share/ai-gateway}"
USER_UNIT_DIR="${HOME}/.config/systemd/user"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

if [[ "${KEYLANE_DEV:-}" == "1" ]] || [[ ! -d "${DEST}/.venv" ]]; then
  DEST="${SRC}"
  say "Using dev tree: ${DEST}"
else
  say "Using install tree: ${DEST}"
fi

chmod +x "${DEST}/scripts/keylane-daemon" "${DEST}/scripts/keylane-ui" "${DEST}/scripts/keylane-toggle"

mkdir -p "${USER_UNIT_DIR}"
for unit in keylane-daemon.service keylane-ui.service; do
  sed "s|%KEYLANE_ROOT%|${DEST}|g" "${SRC}/systemd/${unit}" > "${USER_UNIT_DIR}/${unit}"
done

systemctl --user daemon-reload
systemctl --user enable keylane-daemon.service keylane-ui.service
systemctl --user restart keylane-daemon.service || systemctl --user start keylane-daemon.service

# UI needs a graphical session — start if one is active.
if [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
  systemctl --user restart keylane-ui.service || systemctl --user start keylane-ui.service
else
  say "No graphical session in this shell — UI will start at next login."
fi

say "Enabled on startup:"
systemctl --user is-enabled keylane-daemon.service keylane-ui.service
