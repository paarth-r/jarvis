#!/usr/bin/env bash
# Install the Jarvis voice service: venv, deps, models, and a login service.
# macOS: LaunchAgent (runs in the user session so mic permission applies).
# Linux: systemd user unit. Usage: scripts/install_voice.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
OS="$(uname -s)"
LABEL="com.jarvis.voice"

mkdir -p "$HOME/.jarvis/voice"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
fi

[ -d .venv-voice ] || uv venv --python 3.12 .venv-voice
uv pip install -p .venv-voice -e ".[voice]"

if [ "$OS" = "Linux" ]; then
    # mlx-whisper is Apple-only; faster-whisper is the Linux STT backend.
    # Follow-up: the STT class still needs a faster-whisper implementation.
    uv pip install -p .venv-voice faster-whisper
fi

if [ ! -f scripts/fetch_voice_models.sh ]; then
    echo "scripts/fetch_voice_models.sh not found; cannot fetch models" >&2
    exit 1
fi
bash scripts/fetch_voice_models.sh
for f in models/kokoro/kokoro-v1.0.onnx models/kokoro/voices-v1.0.bin models/smart_turn/smart-turn-v3.2-cpu.onnx; do
    if [ ! -f "$f" ]; then
        echo "Missing model file after fetch: $f" >&2
        exit 1
    fi
done

if [ ! -f config.voice.local.yaml ]; then
    echo "config.voice.local.yaml not found. Copy config.voice.example.yaml to config.voice.local.yaml, set vault and vocab_file, then re-run this script. Service not installed." >&2
    exit 1
fi

if [ "$OS" = "Darwin" ]; then
    PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
    mkdir -p "$HOME/Library/LaunchAgents"
    sed -e "s|__REPO__|$REPO|g" -e "s|__HOME__|$HOME|g" scripts/com.jarvis.voice.plist.template > "$PLIST"
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "Installed LaunchAgent $PLIST"
elif [ "$OS" = "Linux" ]; then
    UNIT_DIR="$HOME/.config/systemd/user"
    mkdir -p "$UNIT_DIR"
    cat > "$UNIT_DIR/jarvis-voice.service" <<UNIT
[Unit]
Description=Jarvis voice agent
After=sound.target network-online.target

[Service]
WorkingDirectory=$REPO
ExecStart=$REPO/.venv-voice/bin/jarvis-voice --config $REPO/config.voice.example.yaml --local $REPO/config.voice.local.yaml
Restart=always
RestartSec=3
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload
    systemctl --user enable --now jarvis-voice.service
    echo "Installed systemd user unit $UNIT_DIR/jarvis-voice.service"
    echo "For start-at-boot without login: sudo loginctl enable-linger $USER"
else
    echo "Unsupported OS: $OS" >&2
    exit 1
fi
