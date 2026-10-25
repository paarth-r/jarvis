# Jarvis

Vision-based human–computer interaction runtime. A stereo camera pair watches your
hands; Jarvis tracks them with MediaPipe, triangulates the landmarks into 3D world
coordinates, and turns gestures into actions — cursor movement, pinch to click or
drag (a quick pinch clicks; holding it past `drag_hold_s` drags until released), and
pointing — with a live 2D + 3D visualizer.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

> Successor to the two-script prototype in `mouse_control/` and `gesture_control/`.
> Those are kept as legacy references; the runtime below is the real system.

---

## How it works

A single asyncio event loop with typed pub/sub events as the spine. Each module
subscribes to event types and publishes new ones:

```
CameraModule ──FrameEvent──▶ HandPoseModule ──PoseEvent──▶ StereoFusionModule
                                                                  │
                                                          WorldPoseEvent (3D, metres)
                                                                  │
                       ┌──────────────────────────┬──────────────┴───────────┐
                  GestureHCIModule           PointingModule              VisualizerModule
                   │ (cursor, pinch)          │ (index-finger ray)        │ (2D overlay + 3D view)
              IntentEvent                  PointEvent ──▶ ActionDispatcher ──▶ OS action
```

- **`WorldPoseEvent` is the stereo boundary.** Everything upstream is per-camera;
  everything downstream works in shared 3D world coordinates.
- **Arm and LLM** run as independent subprocesses over JSON-lines bridges (stubs today).

## Install

Requires Python 3.11+ and macOS (cursor/click use Quartz). From a clone:

```bash
git clone https://github.com/paarth-r/jarvis.git
cd jarvis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `jarvis` console command. The MediaPipe hand model ships in
`mouse_control/hand_landmarker.task`.

## Run

```bash
jarvis --config config.yaml
```

Quit with **Ctrl-C** (or **ESC** in a visualizer window). macOS will prompt for
**Camera** and **Accessibility** permissions on first run — grant them in
System Settings → Privacy & Security.

The default config enables cursor control, so it will move your mouse. To watch
detection without that, enable the visualizer and disable the HCI modules, or use
a replay config.

### No rig? Mono mouse control

`./run_mono.sh` launches mouse control from a single camera (the built-in webcam
by default) using `config.mono.yaml` — StereoFusion mono-projects each hand at a
fixed assumed depth (`mono_depth_m`), so cursor tracking works without the stereo
pair; metric depth (and thus pinch feel) is approximate.

The mono config runs the cursor in **box mode** (`gesture_hci.mode: box`):

- **Hand position is cursor position, from a small box.** About 24 x 15 cm of
  hand travel (`box_width_m`, `box_height_m`) spans the whole screen, linearly,
  with no acceleration.
- **The box comes to you.** When your hand enters view the box is placed so the
  cursor doesn't jump; push past an edge and the box slides with you, so
  reversing comes straight back off the edge.
- **Steady at rest, quick when moving.** A 1-euro filter (`min_cutoff` for
  jitter when holding still, `beta` for lag in fast moves) replaces the fixed EMA.
- **Pinch** clicks / pinch-and-hold drags; the cursor holds its spot while a
  pinch closes so clicks land where you aimed.

Other modes: `absolute` (legacy hand-position mapping via `screen_gain`) and
`trackpad` (relative motion with pointer acceleration; fist = clutch).

### Eye-tracking cursor

`./run_gaze.sh` makes your **eyes** the cursor. The built-in webcam watches both
your face and your hand (`config.gaze.yaml`). The hand never moves the cursor;
it only clicks:

| Gesture | Action |
|---|---|
| look | cursor follows your gaze (1-euro smoothed) |
| thumb + index pinch, quick | left click |
| thumb + index pinch, held > `drag_hold_s` | drag (cursor follows your gaze while held) |
| thumb + middle pinch | right click |

Gaze comes from MediaPipe Face Landmarker: the iris's position inside each eye
plus head pose, mapped to the screen by a per-user calibration. Every launch
starts with calibration, since posture and lighting drift between sessions:
look at 13 dots, then it prints the **held-out error** (leave-one-dot-out, in
screen points) and shows a live dot + halo following your gaze (ESC to start
cursor control). That error is roughly the halo radius, so a click
lands at the halo's centre. Snapping to the nearest control inside the halo
comes later (`mouse_control/click_halo.py` is the starting point).

```bash
./run_gaze.sh                 # calibrate, then run (cursor follows gaze); ESC in calibration aborts
./run_gaze.sh --jump          # cursor stays put; a pinch jumps it to your gaze and clicks
jarvis-gaze-calibrate         # calibration alone -> ~/.jarvis/gaze_calibration.json
```

If you sit differently mid-session, restart to recalibrate. The fit uses head
pose, but a webcam can't fully separate head motion from eye motion.

### No hardware? Replay from a clip

Any camera can read from a recorded video or an image folder instead of a live
device — the whole pipeline (and the 3D viewer) runs with no rig:

```yaml
cameras:
  left:  { source: images, path: /path/to/frames, fps: 30, intrinsics: {...}, extrinsics: {...} }
```

`source` may be `device` (default, live), `video` (a file), or `images` (a folder).

## Voice agent

`jarvis-voice` is an always-on voice front-end for headless Claude Code. It listens
for a wake word ("hey Jarvis"), records until you finish speaking (Silero VAD plus
the smart-turn model, with a 3 s hard stop), transcribes locally with Whisper, sends
the text to a warm Claude Code session running in your vault, and speaks the reply
sentence by sentence with Kokoro TTS. After a reply it keeps listening for 8 s so
you can follow up without the wake word; "new conversation" starts a fresh session,
and "nevermind" (or "cancel", "stop", "forget it") drops the conversation. It cannot
hear you while it is talking (the mic is muted to avoid hearing itself).
A small room MCP server (`list_devices`, `get_state`, `set_state`) lets Claude drive
devices through an HTTP room API. It is separate from the vision runtime and has its
own environment (`.venv-voice`, Python 3.12, Apple silicon for STT today).

### Install

```bash
uv venv --python 3.12 .venv-voice
uv pip install -p .venv-voice -e ".[voice]"
scripts/fetch_voice_models.sh
cp config.voice.example.yaml config.voice.local.yaml
```

Set `vault` (and optionally `vocab_file`, a one-term-per-line list of names Whisper
should spell right) in `config.voice.local.yaml`. Claude Code must be installed and
logged in. `scripts/install_voice.sh` does all of the above and installs a login
service (LaunchAgent on macOS; on Linux the systemd user unit is written but not
enabled until a faster-whisper STT backend lands).

### Run

```bash
.venv-voice/bin/jarvis-room
.venv-voice/bin/jarvis-voice --local config.voice.local.yaml
```

Run the two commands in separate terminals. `jarvis-room` is the real room API: it
exposes only devices that exist, today `speakers` (the Mac's output volume plus the
Spotify desktop app; "play <song>" needs `spotify_client_id` / `spotify_client_secret`
in your local config, Spotify links play without them). Songs, albums, public
playlists and artists can be searched and played, plus skip, shuffle, pause and volume. `jarvis-room-stub` fakes a
fan, window, thermostat and speakers for testing. `jarvis-voice --list-devices` lists audio devices (set `input_device` /
`output_device` in the config), `--no-room` skips the room tools, and `--check`
loads every model and builds the session config, then prints `ok` and exits. State
transitions are logged to stderr as `[voice] IDLE -> LISTEN`.

### Terminal trace

Besides the `[voice]` state changes, the terminal shows `[heard]` (your transcript),
`[thinking]` (Claude's thinking, when it streams any), `[tool]` / `[result]` (each
tool call and its outcome) and `[said]` (each spoken sentence).

### Drop-in wake words

Every `*.onnx` openWakeWord model in `models/wakeword/` is loaded next to the
built-in `hey_jarvis`, and fires the same way. For example,
`scripts/fetch_voice_models.sh --hey-claude` adds a community "hey claude" model.

### Voice

Replies use Kokoro's `bm_daniel`, a British male voice, and address you as "sir".
Change `tts_voice` (any Kokoro voice id; `b*` voices get British pronunciation) or
`address_as` (empty for none) in your local config.

### Confirmation rules

- `permission_mode: auto` (the default) turns on Claude Code's auto mode: its own
  classifier vets every tool call, so shell commands, file edits and lookups just run.
- The voice still asks before anything that reaches other people or can't be undone:
  sending, replying, forwarding or sharing, calendar invites with attendees or event
  updates, comments, and deletes. Texts to your own chat and room control never ask.
- It reads back a short request ("I want to email Sam about lunch. Approve?").
  Obvious answers ("yes", "no") decide instantly; anything looser ("yea approved",
  "eh go for it") is judged by Claude Haiku. Unclear, silence or errors cancel.
- Requests too long to read back (huge recipient lists) are refused outright.
- The rules hold even for tools your Claude Code settings allowlist: they are
  enforced in a PreToolUse hook, which runs before settings allow rules.
- `permission_mode: default` restores the strict gate (shell commands, writes
  outside the vault and most MCP writes ask too).

### Privacy

Personal settings live only in `config.voice.local.yaml` and the vocab file, both
gitignored, as are the model files. Audio is processed on the machine; recorded
utterances (WAV plus a transcript index) are kept under `~/.jarvis/voice`, and the
turn log is written to `wiki/voice-log/` inside your own vault. Nothing personal is
committed. Requests themselves go to Claude through Claude Code.

See [docs/voice/DEPLOY.md](docs/voice/DEPLOY.md) for service deployment and
[docs/voice/room-api.md](docs/voice/room-api.md) for the room API contract.

## Configuration (`config.yaml`)

| Section | Purpose |
|---|---|
| `cameras` | Per-camera `device`/`source`/`path`, `width/height/fps`, `intrinsics` (fx,fy,cx,cy), `distortion` (k1,k2,p1,p2,k3), `extrinsics` (camera-to-world R,t) |
| `hand_pose` | MediaPipe `detection_confidence` (default 0.5) / `tracking_confidence`, applied to every camera |
| `modules` | Declared modules + their settings (stereo fusion, gesture HCI, gaze + gaze HCI, pointing, action dispatcher, visualizer) |
| `subprocesses` | Optional arm / LLM subprocess bridges |

## Modules

| Module | Consumes → Produces |
|---|---|
| `CameraModule` / `FileCameraModule` | device/file → `FrameEvent` |
| `HandPoseModule` | `FrameEvent` → `PoseEvent` (per-camera 2D landmarks) |
| `StereoFusionModule` | `PoseEvent` → `WorldPoseEvent` (3D triangulation, undistortion, mono fallback) |
| `GestureHCIModule` | `WorldPoseEvent` → cursor move (`absolute`, `box` or `trackpad` mode) + pinch press/drag/release, `IntentEvent` |
| `GazeModule` | `FrameEvent` → `GazeEvent` (face + iris landmarks → calibrated, smoothed screen point) |
| `GazeHCIModule` | `GazeEvent` + `WorldPoseEvent` → gaze cursor (`follow` or `jump_on_pinch`), index pinch = left click/drag, middle pinch = right click |
| `PointingModule` | `WorldPoseEvent` → `PointEvent` (index-finger ray) |
| `ActionDispatcher` | `ActionEvent` / `PointEvent` → OS action (with cooldown) |
| `VisualizerModule` | `FrameEvent`/`PoseEvent`/`WorldPoseEvent` → 2D overlay windows + interactive 3D hand viewer |

## Testing

```bash
pytest
```

The suite is fully mocked — no camera or GUI is opened.

## Calibration status

Intrinsics/distortion in `config.yaml` are **placeholders** (reused mono
calibration) pending a charuco calibration of the color stereo pair. The 3D
skeleton shape is correct; absolute metric depth is approximate until then.
Baseline is 0.1524 m (6"); extrinsic rotation is assumed identity until a
`stereo_extrinsics.json` is produced.

## License

MIT — see [LICENSE](LICENSE).
