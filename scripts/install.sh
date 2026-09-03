#!/usr/bin/env bash
# Install Keylane into ~/.local/share/ai-gateway
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${HOME}/.local/share/ai-gateway"
HOTKEY="${KEYLANE_HOTKEY:-<Super>space}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

systemctl --user stop keylane-daemon.service keylane-ui.service 2>/dev/null || true
systemctl --user stop ai-gateway.service ai-launcher.service 2>/dev/null || true

say "Installing system packages"
if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y \
    python3-gobject gtk4 gtk4-layer-shell libnotify portaudio ffmpeg \
    wl-clipboard wmctrl podman podman-compose || true
fi

say "Syncing ${SRC} -> ${DEST}"
mkdir -p "${DEST}"
rsync -a --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude '.git' --exclude 'data' --exclude 'outputs' \
  "${SRC}/" "${DEST}/"

cd "${DEST}"
if [[ ! -d .venv ]]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt

mkdir -p "${DEST}/data"
cp -n config/*.toml "${DEST}/config/" 2>/dev/null || true

say "Installing systemd user units (start on login)"
chmod +x "${DEST}/scripts/keylane-daemon" "${DEST}/scripts/keylane-ui" "${DEST}/scripts/keylane-toggle" "${DEST}/scripts/keylane-mic" "${DEST}/scripts/keylane-settings" "${DEST}/scripts/setup-hotkey.sh"
KEYLANE_DEST="${DEST}" "${DEST}/scripts/enable-startup.sh"

say "Super+Space hotkey"
KEYLANE_DEST="${DEST}" KEYLANE_HOTKEY="${HOTKEY}" "${DEST}/scripts/setup-hotkey.sh"

say "Optional: start SearXNG for web research"
echo "  cd ${DEST}/deploy && podman compose up -d"
if command -v podman >/dev/null 2>&1; then
  # Only if we actually ship a settings file. This used to copy a path that
  # does not exist in the tree, and under `set -e` that aborted the install at
  # its very last step — but only for someone re-installing with SearXNG up,
  # which is why it survived so long.
  if podman ps --format '{{.Names}}' 2>/dev/null | grep -qx keylane-searxng \
     && [[ -f "${DEST}/deploy/searxng/settings.yml" ]]; then
    podman cp "${DEST}/deploy/searxng/settings.yml" keylane-searxng:/etc/searxng/settings.yml
    podman restart keylane-searxng >/dev/null 2>&1 || true
    echo "  SearXNG settings refreshed in keylane-searxng container"
  fi
fi

say "Done. Keylane starts automatically at login."
echo "  Daemon: systemctl --user status keylane-daemon"
echo "  UI:     systemctl --user status keylane-ui"
