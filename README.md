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
