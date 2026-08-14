#!/usr/bin/env bash
# Launch Jarvis mono (single-camera) mouse control using the built-in webcam.
# Usage: ./run_mono.sh            (uses config.mono.yaml next to this script)
#        ./run_mono.sh other.yaml (override the config)
set -euo pipefail

cd "$(dirname "$0")"

CONFIG="${1:-config.mono.yaml}"

if ! command -v jarvis >/dev/null 2>&1; then
  echo "jarvis command not found — install with: pip install -e ." >&2
  exit 1
fi

echo "Starting Jarvis mono mouse control ($CONFIG) — Ctrl-C to quit."
exec jarvis --config "$CONFIG"
