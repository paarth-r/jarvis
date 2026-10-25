# Deploying the voice agent

The service is `jarvis-voice`, an always-on "hey Jarvis" front-end for headless Claude Code. Install with one script on either target; the script picks the right service manager. On Linux it does not enable the service yet (see the Jetson notes).

## Common prerequisites

- Python 3.12 and `uv`.
- Claude Code installed and logged in on the box: run `claude`, then `/login`. The service reuses that login.
- A vault directory the agent can read and write (see the per-target notes below).
- `config.voice.local.yaml` (gitignored) with at least `vault:` and `vocab_file:`. Copy `config.voice.example.yaml` as a starting point.

## Install

```
scripts/install_voice.sh
```

This creates `.venv-voice`, installs `.[voice]`, runs `scripts/fetch_voice_models.sh` (the install stops with an error if that script or any model is missing), and installs the service. Models expected on disk: `models/kokoro/kokoro-v1.0.onnx`, `models/kokoro/voices-v1.0.bin`, `models/smart_turn/smart-turn-v3.2-cpu.onnx`.

Manual run, without the service:

```
.venv-voice/bin/jarvis-voice --config config.voice.example.yaml --local config.voice.local.yaml
```

The room API is `.venv-voice/bin/jarvis-room` (real devices only: the Mac's speakers via Spotify); `.venv-voice/bin/jarvis-room-stub` fakes every device for testing.

## Mac mini (macOS)

- The installer writes `~/Library/LaunchAgents/com.jarvis.voice.plist` from `scripts/com.jarvis.voice.plist.template` and loads it with `KeepAlive`. It is a LaunchAgent, not a daemon, so it runs in the logged-in user session and microphone permission applies.
- Microphone permission: grant it to the venv's python in System Settings > Privacy > Microphone. If it is not granted, audio is silent with no error.
- The vault can be the iCloud-synced Obsidian vault directly.
- Logs: `~/.jarvis/voice/service.log`.
- Restart: `launchctl kickstart -k gui/$(id -u)/com.jarvis.voice`.
- Stop: `launchctl bootout gui/$(id -u)/com.jarvis.voice`.

## Jetson Orin NX (Linux arm64)

- Pending follow-up: Linux STT. Speech-to-text is MLX-only today (Apple silicon), so the service cannot transcribe on Linux until the STT class gains a faster-whisper implementation. The installer adds `faster-whisper` and fetches the models so the box is ready.
- The installer writes a systemd user unit at `~/.config/systemd/user/jarvis-voice.service` but does not enable or start it: with `Restart=always` it would crash-loop until Linux STT lands. Once it does: `systemctl --user enable --now jarvis-voice.service`. To start at boot without an interactive login: `sudo loginctl enable-linger $USER`.
- Logs: `journalctl --user -u jarvis-voice -f`.
- Prerequisite, not implemented here: the vault must be a git clone of a private mirror repository, with pull-before-write and push-after-write. iCloud is not available on Linux. Set up the mirror and clone it before starting the service, then point `vault:` at the clone.
- iMessage is unavailable on Linux, so leave `self_chat_id` null.
- Confirm the audio input device with `arecord -l` and set `input_device` / `output_device` in `config.voice.local.yaml` if the defaults are wrong.

## Measuring accuracy

Corrected utterances are logged to `<data_dir>/utterances.jsonl` (default `~/.jarvis/voice`). After filling in `corrected` on misheard rows:

```
.venv-voice/bin/python scripts/voice_wer.py ~/.jarvis/voice
```

It prints overall word error rate, the count, and the 10 most-missed reference words; add those to the vocab file to prime the recognizer.
