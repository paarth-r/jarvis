#!/usr/bin/env bash
# Launch Jarvis (mono, built-in webcam) and open the gesture studio in the browser.
# Usage: ./run_studio.sh            (uses config.mono.yaml next to this script)
#        ./run_studio.sh other.yaml (override the config)
# The studio lives inside the running Jarvis, so this also turns on hand
# control of the cursor. Ctrl-C quits both.
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="${1:-config.mono.yaml}"

if ! command -v jarvis >/dev/null 2>&1; then
  echo "jarvis command not found — install with: pip install -e ." >&2
  exit 1
fi

# First free port from 8765 up (another local server may already hold 8765).
PORT="${JARVIS_STUDIO_PORT:-8765}"
while lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; do
  PORT=$((PORT + 1))
done
export JARVIS_STUDIO_PORT="$PORT"
URL="http://127.0.0.1:$PORT"

echo "Starting Jarvis ($CONFIG) with the gesture studio at $URL — Ctrl-C to quit."
jarvis --config "$CONFIG" &
JARVIS_PID=$!
trap 'kill -INT "$JARVIS_PID" 2>/dev/null; wait "$JARVIS_PID" 2>/dev/null' INT TERM

# Open the studio once it answers (camera + model take a few seconds).
for _ in $(seq 1 60); do
  if ! kill -0 "$JARVIS_PID" 2>/dev/null; then
    echo "Jarvis exited before the studio came up." >&2
    exit 1
  fi
  if curl -s -o /dev/null "$URL/api/meta"; then
    open "$URL"
    break
  fi
  sleep 0.5
done

wait "$JARVIS_PID"
