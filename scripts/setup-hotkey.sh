#!/usr/bin/env bash
# Bind Super+Space to Keylane toggle (GNOME).
set -euo pipefail

ROOT="${KEYLANE_DEST:-$(cd "$(dirname "$0")/.." && pwd)}"
TOGGLE="${ROOT}/scripts/keylane-toggle"
DESKTOP_SRC="$(dirname "$0")/app.keylane.Toggle.desktop"
DESKTOP_DEST="${HOME}/.local/share/applications/app.keylane.Toggle.desktop"
HOTKEY="${KEYLANE_HOTKEY:-<Super>space}"
MIC_HOTKEY="${KEYLANE_MIC_HOTKEY:-<Ctrl><Shift>m}"
MIC_TOGGLE="${ROOT}/scripts/keylane-mic"
MIC_BINDING_PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/keylane-mic/"

chmod +x "$TOGGLE" "$MIC_TOGGLE"

if ! command -v gsettings >/dev/null 2>&1; then
  echo "gsettings not found — set Super+Space manually to: $TOGGLE"
  exit 1
fi

mkdir -p "${HOME}/.local/share/applications"
sed "s|KEYLANE_ROOT_PLACEHOLDER|${ROOT}|g" "$DESKTOP_SRC" > "$DESKTOP_DEST"
update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true

SCHEMA="org.gnome.settings-daemon.plugins.media-keys"
BINDING_PATH="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/keylane/"

# Remove stale AI Gateway binding that also claimed Super+Space.
gsettings set "${SCHEMA}.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/" binding "" 2>/dev/null || true
gsettings set "${SCHEMA}.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/custom0/" command "" 2>/dev/null || true

# IBUS grabs Super+Space for input-source switching before GNOME media-keys run.
if gsettings list-schemas 2>/dev/null | grep -qx org.freedesktop.ibus.general.hotkey; then
  IBUS_TRIGGERS="$(gsettings get org.freedesktop.ibus.general.hotkey triggers 2>/dev/null || echo '[]')"
  if [[ "${IBUS_TRIGGERS}" == *"<Super>space"* ]]; then
    gsettings set org.freedesktop.ibus.general.hotkey triggers "[]"
    echo "Disabled IBUS Super+Space (was blocking Keylane)"
  fi
fi

# GNOME's own input-source switcher also uses Super+Space by default.
gsettings set org.gnome.desktop.wm.keybindings switch-input-source "[]" 2>/dev/null || true
gsettings set org.gnome.desktop.wm.keybindings switch-input-source-backward "[]" 2>/dev/null || true

# Merge Keylane into existing custom bindings instead of replacing them all.
EXISTING="$(gsettings get "${SCHEMA}" custom-keybindings 2>/dev/null || echo '[]')"
if [[ "${EXISTING}" != *"keylane"* ]]; then
  if [[ "${EXISTING}" == "@as []" || "${EXISTING}" == "[]" ]]; then
    MERGED="['${BINDING_PATH}']"
  else
    MERGED="${EXISTING%]*}, '${BINDING_PATH}']"
  fi
  gsettings set "${SCHEMA}" custom-keybindings "${MERGED}"
fi

gsettings set "${SCHEMA}.custom-keybinding:${BINDING_PATH}" name "Keylane"
gsettings set "${SCHEMA}.custom-keybinding:${BINDING_PATH}" command "$TOGGLE"
gsettings set "${SCHEMA}.custom-keybinding:${BINDING_PATH}" binding "$HOTKEY"

# Microphone toggle (global — shows spotlight if hidden, then toggles mic).
if [[ "${EXISTING}" != *"keylane-mic"* ]]; then
  EXISTING="$(gsettings get "${SCHEMA}" custom-keybindings 2>/dev/null || echo '[]')"
  if [[ "${EXISTING}" == "@as []" || "${EXISTING}" == "[]" ]]; then
    MERGED="['${MIC_BINDING_PATH}']"
  else
    MERGED="${EXISTING%]*}, '${MIC_BINDING_PATH}']"
  fi
  gsettings set "${SCHEMA}" custom-keybindings "${MERGED}"
fi

gsettings set "${SCHEMA}.custom-keybinding:${MIC_BINDING_PATH}" name "Keylane Mic"
gsettings set "${SCHEMA}.custom-keybinding:${MIC_BINDING_PATH}" command "$MIC_TOGGLE"
gsettings set "${SCHEMA}.custom-keybinding:${MIC_BINDING_PATH}" binding "$MIC_HOTKEY"

echo "Super+Space -> $TOGGLE"
echo "${MIC_HOTKEY} -> $MIC_TOGGLE"
echo "Install root: $ROOT"
