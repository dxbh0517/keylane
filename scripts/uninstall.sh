#!/usr/bin/env bash
# Remove Keylane from this machine.
#
#   ./scripts/uninstall.sh            keep models, config and skills
#   ./scripts/uninstall.sh --purge    remove everything, including downloads
#
# Never touches the git checkout you ran it from.
set -euo pipefail

# Both layouts: the current one, and the pre-rename install that may still be
# sitting beside it. Removing one and silently leaving the other is how someone
# ends up with a daemon they cannot find.
BASE="${KEYLANE_HOME:-${HOME}/.local/share/keylane}"
LEGACY="${HOME}/.local/share/ai-gateway"
PURGE=0
ASSUME_YES=0

for arg in "$@"; do
  case "${arg}" in
    --purge) PURGE=1 ;;
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: ${arg}" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

echo "This will remove:"
echo "    services      keylane-daemon.service, keylane-ui.service"
echo "                  (and ai-gateway.service, ai-launcher.service, if present)"
echo "    launcher      desktop entries and icons"
echo "    hotkey        the Keylane keyboard shortcut"
for dir in "${BASE}" "${LEGACY}"; do
  [[ -d "${dir}" ]] || continue
  if [[ ${PURGE} -eq 1 ]]; then
    echo "    everything    ${dir}  (including models, memories and settings)"
  else
    echo "    program       ${dir}, keeping data/"
  fi
done

if [[ ${ASSUME_YES} -eq 0 ]]; then
  read -r -p $'\nContinue? [y/N] ' reply
  [[ "${reply}" =~ ^[yY] ]] || { echo "Cancelled."; exit 0; }
fi

# ------------------------------------------------------------- services ----
say "Stopping services"
systemctl --user disable --now keylane-daemon.service keylane-ui.service 2>/dev/null || true
systemctl --user disable --now ai-gateway.service ai-launcher.service 2>/dev/null || true
for unit in keylane-daemon keylane-ui ai-gateway ai-launcher; do
  rm -f "${HOME}/.config/systemd/user/${unit}.service"
  # Older installs left wants-symlinks in more than one target.
  rm -f "${HOME}"/.config/systemd/user/*.target.wants/"${unit}".service
done
systemctl --user daemon-reload 2>/dev/null || true

# Anything still running from a previous version. pkill would match this
# script's own shell when the pattern appears on its command line, so match
# explicitly and skip ourselves.
stop_matching() {
  local pattern="$1" pid
  for pid in $(pgrep -f -- "${pattern}" 2>/dev/null || true); do
    [[ "${pid}" == "$$" || "${pid}" == "${PPID}" ]] && continue
    kill "${pid}" 2>/dev/null || true
  done
}

stop_matching "keylane/.venv/bin/uvicorn"
stop_matching "daemon\.main"
stop_matching "keylane/current/ui/main.py"
stop_matching "ai-gateway/.venv/bin/uvicorn"
stop_matching "ai-gateway/launcher/main.py"
stop_matching "launcher\.tray"
sleep 1

# ------------------------------------------------------------- desktop -----
say "Removing launcher entries and icons"
rm -f "${HOME}/.local/share/applications/app.keylane.Launcher.desktop" \
      "${HOME}/.local/share/applications/keylane.desktop" \
      "${HOME}/.local/share/applications/ai-gateway-launcher.desktop" \
      "${HOME}/.config/autostart/keylane-panel.desktop"

find "${HOME}/.local/share/icons/hicolor" -name 'keylane*' -delete 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && \
  gtk-update-icon-cache -f "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true
command -v update-desktop-database >/dev/null 2>&1 && \
  update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true

# -------------------------------------------------------------- hotkey -----
if command -v gsettings >/dev/null 2>&1; then
  say "Removing the keyboard shortcut"
  BASE="org.gnome.settings-daemon.plugins.media-keys"
  OURS="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/keylane/"
  EXISTING="$(gsettings get "${BASE}" custom-keybindings 2>/dev/null || echo '[]')"

  KEEP=()
  while read -r path; do
    [[ -z "${path}" ]] && continue
    CMD="$(gsettings get "${BASE}.custom-keybinding:${path}" command 2>/dev/null || echo '')"
    if [[ "${path}" == "${OURS}" || "${CMD}" == *"ai-gateway"* || "${CMD}" == *"eylane"* ]]; then
      gsettings reset-recursively "${BASE}.custom-keybinding:${path}" 2>/dev/null || true
      echo "    removed ${path}"
      continue
    fi
    KEEP+=("'${path}'")
  done < <(printf '%s' "${EXISTING}" | grep -o "'[^']*'" | tr -d "'")

  JOINED="$(IFS=,; printf '%s' "${KEEP[*]:-}")"
  gsettings set "${BASE}" custom-keybindings "[${JOINED}]" 2>/dev/null || true

  # Hand Super+Space back to GNOME's input-source switcher.
  CURRENT="$(gsettings get org.gnome.desktop.wm.keybindings switch-input-source 2>/dev/null || echo '')"
  if [[ "${CURRENT}" == "@as []" || "${CURRENT}" == "[]" ]]; then
    gsettings reset org.gnome.desktop.wm.keybindings switch-input-source 2>/dev/null || true
    gsettings reset org.gnome.desktop.wm.keybindings switch-input-source-backward 2>/dev/null || true
    echo "    restored GNOME's default Super+Space"
  fi

  if gsettings list-schemas 2>/dev/null | grep -qx org.freedesktop.ibus.general.hotkey; then
    IBUS_TRIGGERS="$(gsettings get org.freedesktop.ibus.general.hotkey triggers 2>/dev/null || echo '[]')"
    if [[ "${IBUS_TRIGGERS}" == "@as []" || "${IBUS_TRIGGERS}" == "[]" ]]; then
      gsettings set org.freedesktop.ibus.general.hotkey triggers "['<Super>space']" 2>/dev/null || true
      echo "    restored IBUS Super+Space"
    fi
  fi
fi

# --------------------------------------------------------------- files -----
# Everything irreplaceable lives in data/ now — models, memories, settings,
# skills, themes — so keeping your data is keeping one directory rather than
# picking six out of the tree.
for dir in "${BASE}" "${LEGACY}"; do
  [[ -d "${dir}" ]] || continue
  if [[ ${PURGE} -eq 1 ]]; then
    say "Removing ${dir} entirely"
    rm -rf "${dir}"
  else
    say "Removing the program in ${dir}, keeping your data"
    find "${dir}" -mindepth 1 -maxdepth 1 \
      ! -name data ! -name models ! -name config ! -name skills ! -name themes \
      ! -name outputs ! -name plugins \
      -exec rm -rf {} + 2>/dev/null || true
    echo "    kept: $(ls -1 "${dir}" 2>/dev/null | tr '\n' ' ')"
    echo "    delete it yourself with:  rm -rf ${dir}"
  fi
done

cat <<EOF

$(printf '\033[1m==> Keylane removed.\033[0m')

EOF
if [[ ${PURGE} -eq 0 && -d "${BASE}" ]]; then
  echo "    Your models and settings are still in ${BASE}/data."
  echo "    Re-running scripts/install.sh will pick them up again."
  echo
fi
