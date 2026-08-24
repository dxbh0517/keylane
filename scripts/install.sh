#!/usr/bin/env bash
# Install Keylane into ~/.local/share/ai-gateway and enable the user services.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${HOME}/.local/share/ai-gateway"
HOTKEY="${KEYLANE_HOTKEY:-<Super>space}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

UPDATE=0
[[ "${1:-}" == "--update" ]] && UPDATE=1

if [[ -d "${DEST}" ]]; then
  say "Updating the existing install at ${DEST}"
  # Stop the old version before its files move underneath it.
  systemctl --user stop ai-gateway.service ai-launcher.service 2>/dev/null || true
  pkill -f "ai-gateway/launcher/main.py" 2>/dev/null || true
  pkill -f "launcher.tray" 2>/dev/null || true
fi

# Retire artefacts from older Keylane layouts.
rm -f "${HOME}/.local/share/applications/ai-gateway-launcher.desktop"
# The launcher belongs to graphical-session.target now; an old default.target
# symlink would start it before the desktop exists.
rm -f "${HOME}/.config/systemd/user/default.target.wants/ai-launcher.service"

say "Installing Fedora GTK / PyGObject packages"
if [[ ${UPDATE} -eq 1 ]]; then
  echo "    skipped (--update)"
elif command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y \
    python3-gobject python3-gobject-base \
    gtk4 libadwaita \
    gobject-introspection \
    portaudio ffmpeg || true

  # Tray indicator (GTK 3 AppIndicator) and precise popup placement on wlroots.
  # Either binding works; Fedora ships libappindicator-gtk3.
  sudo dnf install -y libappindicator-gtk3 gnome-shell-extension-appindicator || \
    sudo dnf install -y libayatana-appindicator-gtk3 || \
    echo "    note: no AppIndicator library available — the tray icon will be skipped."
  sudo dnf install -y gtk4-layer-shell || \
    echo "    note: gtk4-layer-shell unavailable — the popup will be centred by the compositor."

  # Optional tools the assistant uses when present.
  sudo dnf install -y wl-clipboard playerctl libnotify || true
fi

say "Copying ${SRC} -> ${DEST}"
mkdir -p "${DEST}"

# --delete keeps the install clean of files removed upstream, so everything the
# *user* owns has to be excluded from the sync entirely. Downloaded weights are
# gigabytes and are not replaceable from this repository.
rsync -a --delete \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.git' \
  --exclude 'outputs' \
  --exclude 'models' \
  --exclude 'config' \
  --exclude 'themes' \
  --exclude 'skills' \
  --exclude 'plugins/community' \
  "${SRC}/" "${DEST}/"

# Seed user-owned directories with anything missing, never overwriting.
# --ignore-existing is what makes an upgrade keep your settings.
for dir in config themes skills plugins/community models; do
  mkdir -p "${DEST}/${dir}"
  [[ -d "${SRC}/${dir}" ]] && \
    rsync -a --ignore-existing "${SRC}/${dir}/" "${DEST}/${dir}/"
done

cd "${DEST}"

# --system-site-packages so the launcher can import Fedora's gi/PyGObject.
if [[ ! -d .venv ]]; then
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [[ ${UPDATE} -eq 1 ]]; then
  echo "    reusing the existing virtualenv (--update)"
else
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
fi

# Drop a .pth so a venv built without --system-site-packages still sees gi.
# sysconfig gives the venv's own site-packages; site.getsitepackages()[0] can
# hand back a system directory we have no business writing to.
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PYVER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${SITE}" == "${DEST}"/* ]]; then
  mkdir -p "${SITE}"
  cat > "${SITE}/_ai_gateway_system_gi.pth" <<EOF
/usr/lib64/python${PYVER}/site-packages
/usr/lib/python${PYVER}/site-packages
EOF
else
  echo "    note: skipping the gi .pth — '${SITE}' is outside the virtualenv."
fi

python -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk; print('PyGObject OK')"
python -m uvicorn --version >/dev/null && echo "uvicorn OK"
python -c "import app.main" >/dev/null && echo "gateway imports OK"

say "Building the handbook"
python scripts/build_docs.py

say "Installing services, launcher entry and icons"
mkdir -p "${HOME}/.config/systemd/user"
cp systemd/ai-gateway.service "${HOME}/.config/systemd/user/"
cp systemd/ai-launcher.service "${HOME}/.config/systemd/user/"

mkdir -p "${HOME}/.local/share/applications"
cp scripts/app.keylane.Launcher.desktop "${HOME}/.local/share/applications/"
cp scripts/app.keylane.Launcher.desktop "${HOME}/.local/share/applications/keylane.desktop"

if [[ -d "${DEST}/assets/icons/hicolor" ]]; then
  mkdir -p "${HOME}/.local/share/icons/hicolor"
  rsync -a "${DEST}/assets/icons/hicolor/" "${HOME}/.local/share/icons/hicolor/"
elif [[ -f "${DEST}/launcher/assets/logo.png" ]]; then
  mkdir -p "${HOME}/.local/share/icons/hicolor/256x256/apps"
  cp "${DEST}/launcher/assets/logo.png" "${HOME}/.local/share/icons/hicolor/256x256/apps/keylane.png"
fi
command -v gtk-update-icon-cache >/dev/null 2>&1 && \
  gtk-update-icon-cache -f "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true
command -v update-desktop-database >/dev/null 2>&1 && \
  update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true

systemctl --user daemon-reload
systemctl --user enable --now ai-gateway.service

# --------------------------------------------------------------- tray host ---
# GNOME has no built-in tray. The AppIndicator extension provides the
# StatusNotifier host the indicator registers with — installed but disabled is
# the common case, and it looks exactly like a broken tray.
if command -v gnome-extensions >/dev/null 2>&1; then
  EXT="appindicatorsupport@rgcjonas.gmail.com"
  if gnome-extensions list 2>/dev/null | grep -q "${EXT}"; then
    if ! gnome-extensions info "${EXT}" 2>/dev/null | grep -qi "Enabled: Yes"; then
      say "Enabling the AppIndicator shell extension (needed for the tray icon)"
      gnome-extensions enable "${EXT}" 2>/dev/null && \
        echo "    enabled ${EXT}" || \
        echo "    could not enable it — do it in the Extensions app"
    fi
  else
    echo "    note: no AppIndicator extension found, so there will be no tray icon."
    echo "          sudo dnf install gnome-shell-extension-appindicator"
  fi
fi

# ---------------------------------------------------------------- hotkey ----
TOGGLE_CMD="${DEST}/.venv/bin/python ${DEST}/launcher/main.py --toggle"

bind_gnome_hotkey() {
  local base="org.gnome.settings-daemon.plugins.media-keys"
  local ours="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/keylane/"

  # 1. Free the accelerator from GNOME's own shortcuts.
  #    Super+Space ships bound to "switch input source".
  local wm current
  for wm in switch-input-source switch-input-source-backward; do
    current="$(gsettings get org.gnome.desktop.wm.keybindings "${wm}" 2>/dev/null || echo '')"
    if [[ "${current}" == *"${HOTKEY}"* ]]; then
      gsettings set org.gnome.desktop.wm.keybindings "${wm}" "[]" || true
      echo "    cleared GNOME's ${HOTKEY} (${wm})"
    fi
  done

  # 2. Walk the existing custom shortcuts. Two entries on one accelerator is
  #    exactly how a working hotkey goes dead, so drop stale Keylane entries
  #    and unbind anything else holding the key.
  local existing
  existing="$(gsettings get "${base}" custom-keybindings 2>/dev/null || echo '[]')"
  local -a current_paths=() keep=()
  # Paths come back as a GVariant array of quoted strings; pull them out
  # literally rather than deleting characters, which would corrupt the paths.
  mapfile -t current_paths < <(printf '%s' "${existing}" | grep -o "'[^']*'" | tr -d "'")

  local path schema bind cmd
  for path in "${current_paths[@]}"; do
    [[ -z "${path}" || "${path}" == "${ours}" ]] && continue
    schema="${base}.custom-keybinding:${path}"
    bind="$(gsettings get "${schema}" binding 2>/dev/null || echo '')"
    cmd="$(gsettings get "${schema}" command 2>/dev/null || echo '')"

    if [[ "${cmd}" == *"ai-gateway"* || "${cmd}" == *"eylane"* ]]; then
      # A Keylane binding from an earlier install — drop it, ours replaces it.
      echo "    removed a previous Keylane shortcut (${path})"
      continue
    fi
    if [[ "${bind}" == *"${HOTKEY}"* ]]; then
      gsettings set "${schema}" binding "''" || true
      echo "    unbound ${HOTKEY} from a conflicting shortcut: ${cmd}"
    fi
    keep+=("'${path}'")
  done

  # 3. Register ours, preserving every unrelated shortcut.
  keep+=("'${ours}'")
  local joined
  joined="$(IFS=,; printf '%s' "${keep[*]}")"
  gsettings set "${base}" custom-keybindings "[${joined}]"

  schema="${base}.custom-keybinding:${ours}"
  gsettings set "${schema}" name    'Keylane'
  gsettings set "${schema}" command "${TOGGLE_CMD}"
  gsettings set "${schema}" binding "${HOTKEY}"
  echo "    bound ${HOTKEY} -> Keylane"
}

if command -v gsettings >/dev/null 2>&1 && [[ "${XDG_CURRENT_DESKTOP:-}" == *GNOME* ]]; then
  say "Keyboard shortcut"
  if [[ "${KEYLANE_BIND_HOTKEY:-ask}" == "yes" ]]; then
    bind_gnome_hotkey
  else
    read -r -p "Bind ${HOTKEY} to open Keylane? [Y/n] " reply
    case "${reply}" in
      [nN]*) echo "    skipped — bind it yourself in Settings → Keyboard" ;;
      *)     bind_gnome_hotkey ;;
    esac
  fi
fi

# ------------------------------------------------------------------ done ----
cat <<EOF

$(printf '\033[1m==> Keylane installed.\033[0m')

    Control panel   http://127.0.0.1:9100/
    Handbook        http://127.0.0.1:9100/docs/
    API explorer    http://127.0.0.1:9100/api-docs

    Popup + tray    systemctl --user enable --now ai-launcher.service
    Or run          ${DEST}/.venv/bin/python ${DEST}/launcher/main.py

    Hotkey          ${HOTKEY}  →  ${TOGGLE_CMD}

Next steps:

  1. Enable the NPU if you have one:
       sudo dnf install intel-npu-driver oneapi-level-zero
       sudo usermod -aG render \$USER      # then log out and back in
       ${DEST}/.venv/bin/python ${DEST}/scripts/check_npu.py

  2. Download a router model:
       Control panel → Models → Search Hugging Face → target "Router"
     Until then the assistant uses keyword matching instead of the NPU.

  3. On GNOME, the tray icon needs an extension:
       sudo dnf install gnome-shell-extension-appindicator
     then log out, back in, and enable it in the Extensions app.

EOF
