#!/usr/bin/env bash
# Download the voice agent's model files into models/ (gitignored).
# Files that already exist are skipped. Usage: scripts/fetch_voice_models.sh [--hey-claude]
#   --hey-claude  also fetch the community "hey claude" wake word into models/wakeword/
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="$REPO/.venv-voice/bin/python"

HEY_CLAUDE=0
for arg in "$@"; do
    case "$arg" in
        --hey-claude) HEY_CLAUDE=1 ;;
        -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

KOKORO_BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
SMART_TURN_URL="https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
HEY_CLAUDE_URL="https://huggingface.co/gdiamos/hey-claude/resolve/main/hey_claude.onnx"

fetch() {
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "have  $dest"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "fetch $dest"
    curl -fL --retry 3 -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

fetch "$KOKORO_BASE/kokoro-v1.0.onnx" models/kokoro/kokoro-v1.0.onnx
fetch "$KOKORO_BASE/voices-v1.0.bin" models/kokoro/voices-v1.0.bin
fetch "$SMART_TURN_URL" models/smart_turn/smart-turn-v3.2-cpu.onnx
mkdir -p models/wakeword
if [ "$HEY_CLAUDE" = 1 ]; then
    fetch "$HEY_CLAUDE_URL" models/wakeword/hey_claude.onnx
fi

if [ ! -x "$PY" ]; then
    echo "$PY not found; create the venv first (scripts/install_voice.sh)" >&2
    exit 1
fi

echo "fetch openWakeWord prebuilt models"
"$PY" -c "import openwakeword.utils as u; u.download_models()"

if [ "$(uname -s)" = "Darwin" ]; then
    echo "fetch Whisper weights (mlx-community/whisper-large-v3-turbo)"
    "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/whisper-large-v3-turbo')"
fi

echo "voice models ready"
