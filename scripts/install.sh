#!/usr/bin/env bash
# Install Keylane into ~/.local/share/keylane, in a layout it can update.
#
# The old install rsynced the tree into ~/.local/share/ai-gateway and excluded
# .git, which meant the installed copy was neither a checkout (so it could not
# pull) nor anything an updater could recognise. It is a release layout now:
#
#   ~/.local/share/keylane/
#     releases/<tag>/        one unpacked version, never modified in place
#     current -> releases/…  what the systemd units point at
#     data/                  memories, models, settings — outside the releases
#     .venv/                 outside too, so a rollback keeps its dependencies
#
# Updating is then a new directory beside the old one and a symlink move, and
# rolling back is the same move in reverse. `data/` is never inside the thing
# being replaced, which is the property that makes any of it safe.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${KEYLANE_HOME:-${HOME}/.local/share/keylane}"
LEGACY="${HOME}/.local/share/ai-gateway"
HOTKEY="${KEYLANE_HOTKEY:-<Super>space}"

# A local install is its own "release". A real update replaces this with the
# tag it downloaded.
RELEASE_TAG="${KEYLANE_RELEASE_TAG:-local}"
RELEASE="${BASE}/releases/${RELEASE_TAG}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }

systemctl --user stop keylane-daemon.service keylane-ui.service 2>/dev/null || true
systemctl --user stop ai-gateway.service ai-launcher.service 2>/dev/null || true

# System packages need root, which means a password prompt. Skip it with
# KEYLANE_SKIP_PACKAGES=1 when they are already installed — a re-install or an
# upgrade does not need to ask again, and sudo has nothing to prompt on when
# this runs from anything but a terminal.
if [[ "${KEYLANE_SKIP_PACKAGES:-0}" != "1" ]] && command -v dnf >/dev/null 2>&1; then
  say "Installing system packages"
  sudo dnf install -y \
    python3-gobject gtk4 gtk4-layer-shell libnotify portaudio ffmpeg \
    wl-clipboard wmctrl podman podman-compose || true
fi

# ── migrate an old install ───────────────────────────────────────────────
# Everything irreplaceable lives in data/, so that is what moves. It is also
# the large thing: models and the compile cache run to tens of gigabytes, and
# a machine with an assistant on it is quite likely not to have that much free
# twice over. So this is a rename when both paths are on one filesystem —
# instant, and it cannot run out of room — and only falls back to a copy when
# they are not, where a copy is the only option anyway.
if [[ -d "${LEGACY}/data" && ! -d "${BASE}/data" ]]; then
  say "Moving your data from ${LEGACY}"
  mkdir -p "${BASE}"
  if [[ "$(stat -c '%d' "${LEGACY}")" == "$(stat -c '%d' "$(dirname "${BASE}")")" ]]; then
    mv "${LEGACY}/data" "${BASE}/data"
    echo "  moved ${LEGACY}/data -> ${BASE}/data"
  else
    need=$(du -sk "${LEGACY}/data" | cut -f1)
    free=$(df -Pk "${BASE}" | awk 'NR==2 {print $4}')
    if (( free < need + 1048576 )); then
      echo "!! ${LEGACY}/data is $((need / 1048576)) GB and only $((free / 1048576)) GB is free" >&2
      echo "   at ${BASE}. Free some space, or move it yourself and re-run." >&2
      exit 1
    fi
    cp -a "${LEGACY}/data" "${BASE}/data"
    echo "  copied ${LEGACY}/data -> ${BASE}/data (different filesystem)"
  fi
  echo "  the old program tree is left at ${LEGACY}; remove it once you are happy"
fi

# The venv comes across too. It is disposable in principle and expensive in
# practice — torch and whisper alone are several gigabytes — so re-creating it
# would mean re-downloading all of that on a machine that may not have room.
# A moved venv has stale absolute paths in its scripts and pyvenv.cfg, which
# `venv --upgrade` rewrites without touching site-packages.
if [[ -d "${LEGACY}/.venv" && ! -d "${BASE}/.venv" ]]; then
  if [[ "$(stat -c '%d' "${LEGACY}")" == "$(stat -c '%d' "$(dirname "${BASE}")")" ]]; then
    say "Reusing the existing virtualenv"
    mkdir -p "${BASE}"
    mv "${LEGACY}/.venv" "${BASE}/.venv"
    python3 -m venv --system-site-packages --upgrade "${BASE}/.venv"
    echo "  moved ${LEGACY}/.venv -> ${BASE}/.venv and repaired its paths"
  fi
fi

say "Installing ${SRC} -> ${RELEASE}"
mkdir -p "${BASE}/releases" "${BASE}/data"
rm -rf "${RELEASE}"
mkdir -p "${RELEASE}"
rsync -a \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude '.git' --exclude 'data' --exclude 'outputs' \
  "${SRC}/" "${RELEASE}/"

# The symlink is what the units follow, so it moves atomically.
ln -sfn "${RELEASE}" "${BASE}/.current.new"
mv -Tf "${BASE}/.current.new" "${BASE}/current"

# The venv sits beside the releases, not inside one: a rollback should not
# have to reinstall its dependencies.
if [[ ! -d "${BASE}/.venv" ]]; then
  python3 -m venv --system-site-packages "${BASE}/.venv"
fi
# `python -m pip` rather than the pip script: a venv that was moved here still
# has the old path baked into its console scripts until they are rewritten,
# and this form never reads them.
"${BASE}/.venv/bin/python" -m pip install -U pip
"${BASE}/.venv/bin/python" -m pip install -r "${BASE}/current/requirements.txt"

DEST="${BASE}/current"

say "Installing systemd user units (start on login)"
chmod +x "${DEST}/scripts/keylane-daemon" "${DEST}/scripts/keylane-ui" \
         "${DEST}/scripts/keylane-toggle" "${DEST}/scripts/keylane-mic" \
         "${DEST}/scripts/keylane-settings" "${DEST}/scripts/keylane-update" \
         "${DEST}/scripts/keylane-status" "${DEST}/scripts/npu-bench.py" \
         "${DEST}/scripts/setup-hotkey.sh"
KEYLANE_DEST="${DEST}" KEYLANE_DATA="${BASE}/data" KEYLANE_VENV="${BASE}/.venv" \
  "${DEST}/scripts/enable-startup.sh"

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
echo "  Daemon:  systemctl --user status keylane-daemon"
echo "  UI:      systemctl --user status keylane-ui"
echo "  Updates: ${DEST}/scripts/keylane-update"
