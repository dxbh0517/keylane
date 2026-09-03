#!/usr/bin/env bash
# Install a matched Intel NPU driver + compiler, and pin OpenVINO to suit.
#
# The NPU compiler is part of the driver, not part of the OpenVINO package.
# Fedora ships the two separately and they have drifted apart: as of writing,
# intel-npu-driver is 1.32.0 (pairs with OpenVINO 2026.0) while
# intel-npu-compiler is 2025.1.0. Nothing compiles for the NPU on that
# combination, and the error names a configuration key rather than the cause.
#
# This installs Intel's own release, where the two halves match, and pins the
# gateway's OpenVINO to the version that release was built against.
#
# Usage:  scripts/npu-driver-fix.sh [--dry-run]
set -euo pipefail

# The driver release to install, and the OpenVINO it was built against. Keep
# these three in step: the point of the script is that they currently are not.
DRIVER_TAG="${KEYLANE_NPU_DRIVER_TAG:-v1.35.0}"
OPENVINO="${KEYLANE_OPENVINO:-2026.2.1}"
GENAI="${KEYLANE_OPENVINO_GENAI:-2026.2.1.0}"
PREFIX="/usr/local/lib64"      # see the ld.so.conf.d drop-in below
LDCONF="/etc/ld.so.conf.d/00-keylane-npu.conf"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# Root escalation has to work from three places: an interactive terminal, a
# desktop session with no controlling TTY (sudo cannot prompt there), and a
# dry run that must touch nothing at all.
ROOT_CMD=""
pick_root() {
  [ "$(id -u)" = 0 ] && { ROOT_CMD=""; return; }
  if sudo -n true 2>/dev/null; then ROOT_CMD="sudo"
  elif [ -t 0 ] && [ -t 1 ]; then ROOT_CMD="sudo"
  elif command -v pkexec >/dev/null 2>&1 && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    # No TTY but a graphical session: pkexec puts up its own dialog.
    ROOT_CMD="pkexec"
  else
    echo "!! Need root, but there is no terminal to ask on and no pkexec." >&2
    echo "   Re-run this script from a normal terminal window." >&2
    exit 1
  fi
}

run()      { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
run_root() { echo "+ $ROOT_CMD $*"; [ "$DRY" = 1 ] || ${ROOT_CMD:+$ROOT_CMD} "$@"; }

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
[ -e /dev/accel/accel0 ] || { echo "No NPU device at /dev/accel/accel0" >&2; exit 1; }

work="$(mktemp -d)"; chmod 755 "$work"; trap 'rm -rf "$work"' EXIT
pick_root
[ -n "$ROOT_CMD" ] && echo "==> Escalating with: $ROOT_CMD"

echo "==> Fetching linux-npu-driver $DRIVER_TAG"
asset=$(curl -fsSL "https://api.github.com/repos/intel/linux-npu-driver/releases/tags/$DRIVER_TAG" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["assets"][0]["name"])')
curl -fsSL -o "$work/rel.tar.gz" \
     "https://github.com/intel/linux-npu-driver/releases/download/$DRIVER_TAG/$asset"
tar xzf "$work/rel.tar.gz" -C "$work"

echo "==> Unpacking the compiler and Level Zero backend"
mkdir -p "$work/root"
for deb in "$work"/*compiler*.deb "$work"/*level-zero-npu_*.deb; do
  [ -e "$deb" ] || continue
  ar p "$deb" data.tar.gz | tar xz -C "$work/root"
done
src="$work/root/usr/lib/x86_64-linux-gnu"
ls "$src"/*.so* >/dev/null 2>&1 || { echo "No libraries unpacked" >&2; exit 1; }

echo "==> Installing into $PREFIX"
# One escalation, not one per file: pkexec puts up a password dialog for every
# invocation, and five dialogs to copy five libraries is a terrible way to ask.
installer="$work/install-root.sh"
{
  echo '#!/bin/sh'
  echo 'set -eu'
  # An earlier version of this script installed into /usr/local/lib, which
  # Fedora's loader does not search. Leaving those behind is just confusion.
  echo "rm -f /usr/local/lib/libze_intel_npu.so* /usr/local/lib/libopenvino_intel_npu_compiler*.so"
  echo "install -d '$PREFIX'"
  # Install only real files; the .so and .so.1 names are recreated as symlinks.
  # ldconfig warns about a versioned soname that is a regular file, and a copy
  # of a 127MB library under three names is wasteful besides.
  for lib in "$src"/*.so*; do
    [ -L "$lib" ] && continue
    echo "install -m 0755 '$lib' '$PREFIX/$(basename "$lib")'"
  done
  for lib in "$src"/*.so*; do
    [ -L "$lib" ] || continue
    echo "ln -sfn '$(basename "$(readlink -f "$lib")")' '$PREFIX/$(basename "$lib")'"
  done
  # Fedora searches neither /usr/local/lib nor /usr/local/lib64 by default, and
  # the drop-in has to sort before the others so these win over Fedora's own
  # copies in /usr/lib64 rather than losing to them.
  echo "printf '%s\\n' '$PREFIX' > '$LDCONF'"
  echo 'ldconfig'
} > "$installer"
chmod 755 "$installer"
echo "--- will run as root ---"; sed 's/^/    /' "$installer"
run_root "$installer"

# Which Keylane to pin. This used to be hardcoded to the pre-rename install
# path, so running the script from a checkout fixed the system libraries and
# left the venv that actually runs on an OpenVINO the new driver does not
# match — which is the exact failure the script exists to prevent.
#   1. KEYLANE_ROOT, if the caller set it.
#   2. The checkout this script is in, if it has a venv.
#   3. The installed copy.
pick_root_dir() {
  if [ -n "${KEYLANE_ROOT:-}" ] && [ -x "${KEYLANE_ROOT}/.venv/bin/pip" ]; then
    echo "${KEYLANE_ROOT}"; return
  fi
  here="$(cd "$(dirname "$0")/.." && pwd)"
  if [ -x "${here}/.venv/bin/pip" ]; then echo "${here}"; return; fi
  for candidate in "$HOME/.local/share/keylane/current" \
                   "$HOME/.local/share/keylane" \
                   "$HOME/.local/share/ai-gateway"; do
    [ -x "${candidate}/.venv/bin/pip" ] && { echo "${candidate}"; return; }
  done
  echo ""
}

GW="$(pick_root_dir)"
echo "==> Pinning OpenVINO to $OPENVINO in ${GW:-<no venv found>}"
if [ -n "$GW" ]; then
  # `python -m pip`, not the pip script. A venv that was moved into place —
  # which is exactly what install.sh does when it migrates an old install —
  # keeps the old path in its console-script shebangs, so calling bin/pip
  # directly dies with "bad interpreter" on a machine that is otherwise fine.
  run "$GW/.venv/bin/python" -m pip install --quiet \
      "openvino==$OPENVINO" "openvino-genai==$GENAI"
else
  echo "!! Found no Keylane venv. Pin it yourself:" >&2
  echo "     pip install openvino==$OPENVINO openvino-genai==$GENAI" >&2
fi

echo "==> Restarting Keylane"
# The units were renamed; the old name was still here and silently did nothing.
for unit in keylane-daemon.service keylane-ui.service; do
  if systemctl --user list-unit-files "$unit" >/dev/null 2>&1; then
    run systemctl --user try-restart "$unit"
  fi
done

cat <<'EOF'

Done. Check the Models page, or:
  curl -s http://127.0.0.1:9100/health | python3 -m json.tool

Then measure it, rather than guessing whether it helped:
  PYTHONPATH=. python scripts/npu-bench.py

A cold 7B compile should come out in single-digit seconds — 6.8 s on an
NPU 3720. A minute or more means the driver and the compiler are still out of
step. This does not change how fast the model *answers*: on the same machine
time to first token stayed at ~15 s and decode at ~0.25 s/token before and
after. What it fixes is compile time, and VLM pipelines, which throw
ZE_RESULT_ERROR_UNINITIALIZED on a mismatched stack. To undo, remove the
libraries, run ldconfig, and reinstall the OpenVINO version you had:

  sudo rm -f /usr/local/lib64/lib*npu*.so* /etc/ld.so.conf.d/00-keylane-npu.conf
  sudo ldconfig
EOF
