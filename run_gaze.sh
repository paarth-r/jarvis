#!/usr/bin/env bash
# Launch Jarvis eye-tracking cursor control from the built-in webcam.
# Usage: ./run_gaze.sh                (calibrates first if there's no calibration yet)
#        ./run_gaze.sh --recalibrate  (always calibrate first)
#        ./run_gaze.sh --jump         (cursor jumps to your gaze only when you pinch)
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="config.gaze.yaml"
CALIB="$HOME/.jarvis/gaze_calibration.json"
RECALIBRATE=0
JUMP=0
for arg in "$@"; do
  case "$arg" in
    --recalibrate) RECALIBRATE=1 ;;
    --jump) JUMP=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-python3}"
if [[ $RECALIBRATE == 1 || ! -f "$CALIB" ]]; then
  "$PY" -m jarvis.gaze_calibrate --out "$CALIB"
fi

if [[ $JUMP == 1 ]]; then
  TMP="$(mktemp -t jarvis-gaze).yaml"
  sed 's/jump_on_pinch: false/jump_on_pinch: true/' "$CONFIG" > "$TMP"
  CONFIG="$TMP"
fi

echo "Starting Jarvis gaze control ($CONFIG) - Ctrl-C to quit."
exec "$PY" -m jarvis.main --config "$CONFIG"
