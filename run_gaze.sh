#!/usr/bin/env bash
# Launch Jarvis eye-tracking cursor control from the built-in webcam.
# Calibrates every launch (posture/lighting drift between sessions), then runs.
# Usage: ./run_gaze.sh
#        ./run_gaze.sh --jump         (cursor jumps to your gaze only when you pinch)
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="config.gaze.yaml"
CALIB="$HOME/.jarvis/gaze_calibration.json"
JUMP=0
for arg in "$@"; do
  case "$arg" in
    --jump) JUMP=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Use the interpreter the `jarvis` command was installed into (has mediapipe + Quartz).
if [[ -z "${PYTHON:-}" ]] && command -v jarvis >/dev/null 2>&1; then
  PYTHON="$(head -1 "$(command -v jarvis)" | sed 's/^#!//')"
fi
PY="${PYTHON:-python3}"
# Cancelling calibration (ESC) exits non-zero, which stops the launch here.
"$PY" -m jarvis.gaze_calibrate --out "$CALIB"

if [[ $JUMP == 1 ]]; then
  TMP="$(mktemp -t jarvis-gaze).yaml"
  sed 's/jump_on_pinch: false/jump_on_pinch: true/' "$CONFIG" > "$TMP"
  CONFIG="$TMP"
fi

echo "Starting Jarvis gaze control ($CONFIG) - Ctrl-C to quit."
exec "$PY" -m jarvis.main --config "$CONFIG"
