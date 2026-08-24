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

DRIVER_TAG="v1.35.0"          # pairs with OpenVINO 2026.2
OPENVINO="2026.2.1"
GENAI="2026.2.1.0"
PREFIX="/usr/local/lib"        # ahead of /usr/lib64 for the dynamic loader
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
[ -e /dev/accel/accel0 ] || { echo "No NPU device at /dev/accel/accel0" >&2; exit 1; }

work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
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
run sudo install -d "$PREFIX"
for lib in "$src"/*.so*; do run sudo install -m 0755 "$lib" "$PREFIX/$(basename "$lib")"; done
run sudo ldconfig

echo "==> Pinning the gateway's OpenVINO to $OPENVINO"
GW="${KEYLANE_HOME:-$HOME/.local/share/ai-gateway}"
if [ -x "$GW/.venv/bin/pip" ]; then
  run "$GW/.venv/bin/pip" install --quiet \
      "openvino==$OPENVINO" "openvino-genai==$GENAI"
else
  echo "!! No venv at $GW/.venv — pin OpenVINO to $OPENVINO yourself." >&2
fi

echo "==> Restarting the gateway"
run systemctl --user restart ai-gateway.service

cat <<'EOF'

Done. Check the Models page, or:
  curl -s http://127.0.0.1:9100/api/status | python3 -m json.tool | grep assistant

The first NPU load compiles the model and can take several minutes; it is
cached afterwards. To undo, remove the libraries from /usr/local/lib, run
ldconfig, and reinstall the OpenVINO version you had.
EOF
