#!/usr/bin/env bash
# Launch the Jarvis room API (Mac speakers + Spotify) that the voice agent drives.
# Usage: ./run_room.sh                        (uses config.voice.local.yaml next to this script)
#        ./run_room.sh other.yaml             (override the local config)
#        ./run_room.sh other.yaml --port 8791 (extra args go to jarvis-room)
# Start this before ./run_voice.sh, in its own terminal. Ctrl-C quits.
set -euo pipefail

cd "$(dirname "$0")"

BIN=".venv-voice/bin/jarvis-room"
LOCAL="${1:-config.voice.local.yaml}"
shift || true

if [[ ! -x "$BIN" ]]; then
  echo "$BIN not found — install with: scripts/install_voice.sh" >&2
  exit 1
fi
if [[ ! -f "$LOCAL" ]]; then
  echo "$LOCAL not found — cp config.voice.example.yaml $LOCAL and fill it in" >&2
  exit 1
fi

echo "Starting the room API ($LOCAL) — Ctrl-C to quit."
exec "$BIN" --local "$LOCAL" "$@"
