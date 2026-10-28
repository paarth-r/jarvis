#!/usr/bin/env bash
# Launch the Jarvis voice agent ("hey Jarvis" -> headless Claude Code -> spoken reply).
# Usage: ./run_voice.sh                    (uses config.voice.local.yaml next to this script)
#        ./run_voice.sh other.yaml         (override the local config)
#        ./run_voice.sh other.yaml --check (extra args go to jarvis-voice)
# Music and other room tools need ./run_room.sh running first; without it the
# agent still talks, but cannot play anything. Ctrl-C quits.
set -euo pipefail

cd "$(dirname "$0")"

BIN=".venv-voice/bin/jarvis-voice"
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

PORT="${JARVIS_ROOM_PORT:-8790}"
if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "note: nothing on :$PORT — start ./run_room.sh for music and room control." >&2
fi

echo "Starting Jarvis voice ($LOCAL) — Ctrl-C to quit."
exec "$BIN" --local "$LOCAL" "$@"
