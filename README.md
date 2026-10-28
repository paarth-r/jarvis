# Jarvis

Vision-based human–computer interaction runtime. A stereo camera pair watches your
hands; Jarvis tracks them with MediaPipe, triangulates the landmarks into 3D world
coordinates, and turns gestures into actions — cursor movement, pinch to click or
drag (a quick pinch clicks; holding it past `drag_hold_s` drags until released),
fist-then-tilt to scroll, and your own custom gestures bound to media keys, key
combos or macros — with a live 2D + 3D visualizer and a browser-based gesture studio.

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
           ┌──────────────────────┬───────────────┼────────────────┬───────────────┐
    GestureHCIModule        PointingModule   StudioModule     VisualizerModule
     │ cursor, pinch,        (index ray)      (web UI, :8765)   (2D + 3D view)
     │ gesture engine, scroll
  GestureEvent ──▶ ActionDispatcher ──▶ media key / key combo / macro / shell
```

- **`WorldPoseEvent` is the stereo boundary.** Everything upstream is per-camera;
  everything downstream works in shared 3D world coordinates.
- **One gesture at a time.** `GestureHCIModule` runs a gesture engine that picks
  a single active gesture per frame (see [Gestures](#gestures)); the dispatcher
  only ever sees its arbitrated start/end events.
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

## Gestures

| Gesture | What it does |
|---|---|
| pinch (thumb to index) | quick = click, held past `drag_hold_s` = drag |
| fist, then tilt | scroll: tip your knuckles toward the camera to scroll down, back to scroll up |
| point (index out) | bindable; Cmd+Space by default |
| your own | whatever you bind: media key, key combo, typed text, shell command, macro |

**One gesture at a time.** Every frame the hand shape yields at most one
candidate (fist, then a custom gesture, then point). A candidate has to be held
for a short dwell before it counts (fist 0.1 s, point 0.3 s, custom 0.15 s), and
an active gesture survives tracking dropouts up to 0.1 s. Poses your hand only
passes through never fire: the index straightening for a frame on the way into
a fist is not a point. A fist-shaped hand locks pinch out from its first frame
and holds the cursor, and pinch re-arms only after the thumb opens, so closing
or opening a fist never clicks or drags. Pinch itself keeps its old latency.

**Fist-tilt scroll.** Tilt is measured from the angle your fist had when it
closed, so you can start from any wrist angle. Speed grows with tilt past a dead
zone (rate control, like a joystick). Tune it under `gesture_hci.scroll`:
`deadzone_deg` (8), `full_deg` (35), `max_px_s` (2500), `curve` (1.5),
`smoothing`, `invert_y`, and `horizontal` (turning the forearm scrolls sideways;
off by default).

Shape and tilt use MediaPipe's metric hand landmarks (`WorldPoseEvent.hand_shape`),
so they work from one webcam; the mono cursor path is unchanged.

### Gesture studio

```bash
./run_studio.sh             # mono Jarvis + opens the studio in your browser; Ctrl-C quits
./run_studio.sh other.yaml  # same, with another config
```

The studio lives inside the running Jarvis (so hand control of the cursor is on
too), at **http://127.0.0.1:8765**; if something else holds that port,
`run_studio.sh` takes the next free one and opens that instead
(`JARVIS_STUDIO_PORT` sets it by hand). On the left, your tracked
hand in 3D (drag to orbit), the active gesture, how closely your hand matches each
custom gesture, and the tilt gauge while you scroll. On the right, every gesture
with its binding.

**Add a gesture** in five steps:

1. Name it.
2. Record takes (button or Space): a 3 s countdown, then hold for 1.5 s. Five
   takes at slightly different hand angles is a good set.
3. Optionally record 10 s of everyday hand movement, so the check can catch a
   gesture you'd make by accident.
4. Check it (Enter). Three verdicts, each pass / warn / fail with a hint:
   - **Recognizable**: the tracker kept your hand (few dropouts) and held it
     steadily (hidden fingers make it guess).
   - **Repeatable**: your takes agree. An odd take is named and left out.
   - **Distinct**: far enough from every saved gesture (it names the nearest and
     the fingers that tell them apart), not read as fist or pinch (reserved),
     a warning if it looks like point (it takes over point while held), and not
     matching your everyday movement.
5. Bind and save. A fail blocks saving; a warn doesn't.

Custom gestures need no training step: they match your saved takes by nearest
neighbour (few-shot, like the $1 recognizer), with an acceptance radius fitted to
how much your takes vary, so anything outside every radius is no gesture. A
gesture works the moment it's saved. While a take records, cursor control and
gesture actions pause.

**Bindings** cover media keys (play/pause, next, previous, volume, mute), key
combos (press them in the field, or type `cmd+shift+4`), typed text, shell
commands, and macros (steps of those plus waits). "Repeat while held" re-fires
every N seconds (volume up held keeps turning it up). "Nothing" turns a default
off. "Try it" runs a binding right away.

Gestures and bindings live in `~/.jarvis/gestures.json` (override with
`JARVIS_GESTURES_PATH`); the runtime picks up changes live. The studio listens on
localhost only and refuses requests from other websites, since a binding can run
a shell command.

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
the smart-turn model, with a 3 s hard stop; if the transcript trails off on "uh",
"like", "and" (with no full stop or question mark after it) or a dash it keeps listening
for the rest), transcribes locally with Whisper, sends
the text to a warm Claude Code session running in your vault, and speaks the reply
sentence by sentence with Kokoro TTS. After a reply it keeps listening for 8 s so
you can follow up without the wake word; "new conversation" starts a fresh session,
and "nevermind" (or "cancel", "stop", "forget it") drops the conversation. A goodbye ("thanks", "sounds good",
"that's all") ends the follow-up too, unless it answers a question it just asked. While it
is talking you can cut it off with "wait" or "hold up" (see Barge-in below); otherwise the
mic is muted while it speaks so it never hears itself.
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
Spotify desktop app). Songs, albums, public playlists, artists, your own playlists and
your Liked Songs can be searched and played, plus skip, shuffle, pause and volume.
Setup: put `spotify_client_id` in your local config (a developer.spotify.com app with
redirect URI `http://127.0.0.1:8888/callback`), then run `jarvis-spotify-login` once;
the token is cached in `~/.jarvis/` and refreshes itself. A `spotify_client_secret` is
optional (catalog search without the login). `jarvis-room-stub` fakes a
fan, window, thermostat and speakers for testing. `jarvis-voice --list-devices` lists audio devices (set `input_device` /
`output_device` in the config), `--no-room` skips the room tools, and `--check`
loads every model and builds the session config, then prints `ok` and exits. State
transitions are logged to stderr as `[voice] IDLE -> LISTEN`.

### Spotify from the command line

`jarvis-dj` drives the same speakers and library without the room server, printing
JSON, so any script or Claude session can DJ: `jarvis-dj play "gym" --kind my_playlist`,
`jarvis-dj play --kind liked`, `jarvis-dj top --range short`, `jarvis-dj recent`,
`jarvis-dj playlists`, `jarvis-dj search "<q>" --kind artist`, `pause`, `resume`,
`next`, `prev`, `shuffle on`, `volume 40`, `now`. There is no queue: the desktop app's
AppleScript can only start a track, album, playlist or artist.

### Terminal trace

Besides the `[voice]` state changes, the terminal shows `[heard]` (your transcript),
`[thinking]` (Claude's thinking, when it streams any), `[tool]` / `[result]` (each
tool call and its outcome), `[said]` (each spoken sentence), `[interrupt]` (the stop
phrase that cut a reply off) and `[turn]` (why your
turn ended: `smart_turn`, `hard_silence` or `max_length`, with the pause length and
the smart-turn score).

### Drop-in wake words

Every `*.onnx` openWakeWord model in `models/wakeword/` is loaded next to the
built-in `hey_jarvis`, and fires the same way. For example,
`scripts/fetch_voice_models.sh --hey-claude` adds a community "hey claude" model.

### Voice

Replies use Kokoro's `bm_daniel`, a British male voice, and address you as "sir".
Change `tts_voice` (any Kokoro voice id; `b*` voices get British pronunciation) or
`address_as` (empty for none) in your local config.

### Music ducking

While Jarvis is engaged (from the wake word until the conversation ends) it lowers
Spotify and Apple Music to `duck_ratio` (default 0.2) of their current level, then
fades them back to exactly where they were. Only each app's own volume is changed,
never the system volume Jarvis speaks at, and only apps that are already running.
If you change an app's volume while it is ducked, your choice is kept. Levels are
saved to `~/.jarvis/voice/duck.json` while ducked, so a crash cannot leave music
low: the next start restores them. Music started mid-conversation is lowered before
the next sentence, but a hung or launching app never delays speech by more than about
0.3 s, and an app that fails is left alone until the conversation ends. Set
`duck_music: false` to turn it off. macOS has
no per-app volume API, so only Spotify and Apple Music can be ducked.

### Barge-in

Say "wait", "no", "hold up", "hang on", "hold on" or "jarvis" while Jarvis is reading
a reply and it stops at once, drops the rest of Claude's answer without losing the
Claude session, and listens for what you say next. That next turn goes to Claude with a
note of roughly where it was cut off, based on how much of the sentence actually played.
A stop phrase on its own ("wait", "hold up") just stops it and leaves it listening; "stop"
or "nevermind" still ends the conversation with "Okay.".

There is no echo cancellation: the mic listens while Jarvis speaks, so a phrase counts
only if it is not in the sentence being spoken (Jarvis saying "no problem" cannot stop
itself). The mic also hears the words Jarvis played just before you cut in, so the
transcript of what you say next is trimmed to start at your stop phrase ("...sunny and
wait, play jazz" reaches Claude as "wait, play jazz"). Detection is word-based: a
separate Whisper (no vocabulary prompt, so it never "hears" a prompt word like Jarvis in
noise) transcribes the last two seconds every quarter second, and a phrase must show up
in two windows in a row (a window cut mid-word can turn "nothing" into "no idea"), so
expect about three quarters of a second before it stops. Only reply sentences can be interrupted: confirmation
questions, tones and error messages still play with the mic muted, and interrupting
while Claude is still thinking (before it starts speaking) is not covered yet. Set
`barge_in: false` to keep the mic muted for all playback, or change `stop_phrases`.

`scripts/bargein_soak.py` checks for false barge-ins offline: it synthesizes 22 replies
full of near-misses ("now", "know", "nothing", "wait time", "on hold", "Jarvis") with
your Kokoro voice and feeds them to the real listener; a hit means Jarvis stopped itself.
It reported 0 hits (1 with `--single-window`) clean, with noise and at low gain. It feeds
the TTS audio in directly, so it does not model the room: speaker and mic coloring,
reverb, or music playing in the background.

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
| `modules` | Declared modules + their settings (stereo fusion, gesture HCI incl. `gestures` dwell and `scroll`, gaze + gaze HCI, pointing, action dispatcher incl. `bindings`, studio `port`, visualizer) |
| `subprocesses` | Optional arm / LLM subprocess bridges |

## Modules

| Module | Consumes → Produces |
|---|---|
| `CameraModule` / `FileCameraModule` | device/file → `FrameEvent` |
| `HandPoseModule` | `FrameEvent` → `PoseEvent` (per-camera 2D landmarks) |
| `StereoFusionModule` | `PoseEvent` → `WorldPoseEvent` (3D triangulation, undistortion, mono fallback) |
| `GestureHCIModule` | `WorldPoseEvent` → cursor move (`absolute`, `box` or `trackpad` mode) + pinch press/drag/release, fist-tilt scroll, `IntentEvent`; gesture engine → `GestureEvent` (start/end) + `GestureStateEvent` (every frame) |
| `GazeModule` | `FrameEvent` → `GazeEvent` (face + iris landmarks → calibrated, smoothed screen point) |
| `GazeHCIModule` | `GazeEvent` + `WorldPoseEvent` → gaze cursor (`follow` or `jump_on_pinch`), index pinch = left click/drag, middle pinch = right click |
| `PointingModule` | `WorldPoseEvent` → `PointEvent` (index-finger ray) |
| `ActionDispatcher` | `GestureEvent` / `ActionEvent` → bound action (media, keys, text, shell, macro), per-gesture cooldown, repeat while held |
| `StudioModule` | `WorldPoseEvent` + `GestureStateEvent` → gesture studio web UI; `StudioModeEvent` pauses cursor and actions while recording |
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
