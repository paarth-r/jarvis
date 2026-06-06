# Jarvis

Vision-based human–computer interaction runtime. A stereo camera pair watches your
hands; Jarvis tracks them with MediaPipe, triangulates the landmarks into 3D world
coordinates, and turns gestures into actions — cursor movement, pinch-to-click, and
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
| `modules` | Declared modules + their settings (stereo fusion, gesture HCI, pointing, action dispatcher, visualizer) |
| `subprocesses` | Optional arm / LLM subprocess bridges |

## Modules

| Module | Consumes → Produces |
|---|---|
| `CameraModule` / `FileCameraModule` | device/file → `FrameEvent` |
| `HandPoseModule` | `FrameEvent` → `PoseEvent` (per-camera 2D landmarks) |
| `StereoFusionModule` | `PoseEvent` → `WorldPoseEvent` (3D triangulation, undistortion, mono fallback) |
| `GestureHCIModule` | `WorldPoseEvent` → cursor move + pinch-click, `IntentEvent` |
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
