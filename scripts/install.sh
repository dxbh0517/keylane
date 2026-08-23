#!/usr/bin/env bash
# Install Keylane into ~/.local/share/ai-gateway and enable user services.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${HOME}/.local/share/ai-gateway"

echo "==> Ensuring Fedora GTK / PyGObject packages"
if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y \
    python3-gobject python3-gobject-base \
    gtk4 libadwaita \
    gobject-introspection \
    portaudio ffmpeg || true
fi

echo "==> Installing from ${SRC} -> ${DEST}"
mkdir -p "${DEST}"
rsync -a --delete \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'outputs' \
  --exclude '.git' \
  "${SRC}/" "${DEST}/"

cd "${DEST}"
# --system-site-packages so the launcher can import Fedora's gi/PyGObject
if [[ ! -d .venv ]]; then
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Also drop a .pth so existing venvs without --system-site-packages still see gi.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
mkdir -p "${SITE}"
cat > "${SITE}/_ai_gateway_system_gi.pth" <<EOF
/usr/lib64/python${PYVER}/site-packages
/usr/lib/python${PYVER}/site-packages
EOF

python -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk; print('PyGObject OK')"

mkdir -p "${HOME}/.config/systemd/user"
cp systemd/ai-gateway.service "${HOME}/.config/systemd/user/"
cp systemd/ai-launcher.service "${HOME}/.config/systemd/user/"

mkdir -p "${HOME}/.local/share/applications"
cp scripts/ai-gateway-launcher.desktop "${HOME}/.local/share/applications/"

systemctl --user daemon-reload
systemctl --user enable --now ai-gateway.service

echo
# Desktop icon for GNOME/KDE
mkdir -p "${HOME}/.local/share/icons/hicolor/256x256/apps"
cp "${DEST}/launcher/assets/logo.png" "${HOME}/.local/share/icons/hicolor/256x256/apps/keylane.png" 2>/dev/null || true
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true
fi

echo "==> Keylane installed."
echo "    Control panel:  http://127.0.0.1:9100/"
echo "    Start launcher: systemctl --user enable --now ai-launcher.service"
echo "    Or run:         ${DEST}/.venv/bin/python ${DEST}/launcher/main.py"
echo "    API:            http://127.0.0.1:9100/api/status"
echo
echo "Bind a GNOME shortcut (Settings → Keyboard → Custom Shortcuts):"
echo "  Name: Keylane"
echo "  Command: ${DEST}/.venv/bin/python ${DEST}/launcher/main.py"
echo "  Shortcut: Super+Space (fallback if bare Super is reserved)"
echo
echo "Next: place an OpenVINO GenAI model under ${DEST}/models/router"
echo "      then run: scripts/check_npu.py"
