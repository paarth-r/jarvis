# Jarvis Hand Visualizer — Design Spec

**Date:** 2026-06-02
**Status:** Approved (pending spec review)

## Goal

Let a developer *see* the system working: per-camera 2D hand-landmark overlays
(proving detection) and an interactive 3D hand-skeleton view of the triangulated
landmarks (proving stereo fusion). Modeled on Hyperform's body-pose `viewer_3d.py`.

## Background

The Jarvis runtime already produces:
- `PoseEvent` — per-camera 2D landmarks (21×3 normalized), one per detected hand frame.
- `WorldPoseEvent` — triangulated 3D landmarks (21×3 metres) in the world frame.

There is currently a placeholder `DebugOverlayModule` whose 2D rendering
back-projects the 3D pose with a crude approximation. This spec **supersedes and
removes** `DebugOverlayModule` in favor of an accurate visualizer.

## Constraints

- **macOS GUI threading.** matplotlib's interactive window needs its own GUI
  event loop; running it in-process alongside the asyncio loop is unsafe.
  Hyperform solves this by running the 3D viewer in a spawned subprocess. cv2
  HighGUI works from the main thread (the prototype uses it), and Jarvis's
  asyncio loop runs on the main thread, so cv2 calls from inside a coroutine are
  main-thread-safe.
- **No GUI in tests.** Unit tests must not open windows or spawn the matplotlib
  subprocess (mirrors how camera/landmarker are mocked elsewhere).

## Architecture

Approach: **matplotlib 3D in a subprocess + cv2 2D overlays on the main thread.**
The 3D IPC payload is just 21 points per frame (tiny), not camera frames.

### Component 1 — `jarvis/modules/viewer_3d.py` : `Viewer3D`

Ported from Hyperform `tools/viewer_3d.py`, adapted to hands.

- Spawns a subprocess (`multiprocessing.get_context("spawn")`) owning a
  matplotlib 3D figure; fed by a `multiprocessing.Queue(maxsize=2)`.
- Worker drains the queue keeping only the freshest frame, redraws at ~30 fps via
  `plt.pause`, preserves the user's drag-rotate angle across redraws, equal aspect
  ratio, dark theme.
- Draws the **21-landmark MediaPipe hand skeleton** using the hand connection
  list (the same connections currently in `debug_overlay._CONNECTIONS`).
- **Coordinate mapping:** Jarvis world frame is OpenCV-style (X right, Y down,
  Z forward, origin at baseline midpoint). The viewer displays
  `(X_disp, Y_disp, Z_disp) = (x, z, -y)` → (right, depth, up) for an intuitive
  upright view, matching Hyperform's convention.
- Public API: `start()`, `update(points)`, `close()`. `update` is non-blocking
  and drops a stale queued frame. `close()` sends a `None` sentinel.
- Payload `points`: a list of 21 items, each `(x, y, z)` tuple or `None`.

### Component 2 — `jarvis/modules/visualizer.py` : `VisualizerModule(Module)`

- **Config:** `enabled` (bool), `show_2d` (bool), `show_3d` (bool), `fps` (int,
  default 30), plus injected `cameras`.
- **Subscriptions** (cache latest, never block publishers):
  - `FrameEvent` → `self._frames[camera_id] = frame`
  - `PoseEvent` → `self._poses_2d[camera_id] = landmarks_2d`
  - `WorldPoseEvent` → `self._world = landmarks_3d` (+ confidence)
- **setup():** if `show_3d`, construct and `start()` a `Viewer3D`.
- **run():** loop at `fps` on the main thread:
  - if `show_2d`: for each camera, draw cached 2D landmarks on the cached frame
    (`draw_landmarks`) and `cv2.imshow(f"jarvis {camera_id}", img)`.
  - if `show_3d`: `viewer.update(hand_points(self._world))`.
  - `cv2.waitKey(1)`; ESC breaks the loop.
  - `await asyncio.sleep(1/fps)`.
  - If neither `show_2d` nor `show_3d`: idle (`await asyncio.sleep(3600)`).
- **teardown():** `cv2.destroyAllWindows()`; `viewer.close()` if started.

### Pure helpers (unit-tested, no GUI)

- `hand_points(world_landmarks: np.ndarray | None) -> list[tuple|None]`
  Converts a (21,3) world array to the viewer's display-space point list;
  returns `[None]*21` when input is `None`.
- `draw_landmarks(frame: np.ndarray, lm2d: np.ndarray, connections) -> np.ndarray`
  Returns an annotated BGR image (handles mono input via GRAY2BGR); draws bones
  and joints. Pure given inputs.

### Hand skeleton connections

The 21-point MediaPipe connection list (thumb/index/middle/ring/pinky chains +
palm), relocated to a shared constant reused by both the 2D overlay and the 3D
viewer.

## Config change

Replace the `debug_overlay` module block with:

```yaml
  visualizer:
    class: jarvis.modules.visualizer.VisualizerModule
    enabled: false        # opt-in; set true to show windows
    show_2d: true
    show_3d: true
    fps: 30
```

Remove `jarvis/modules/debug_overlay.py` and `tests/test_debug_overlay.py`.

## Testing

- `hand_points`: maps a known (21,3) array correctly (axis swap x,z,-y); `None`
  → `[None]*21`.
- `draw_landmarks`: returns an ndarray with the input H×W and 3 channels; accepts
  both mono (2D) and BGR (3D) frames; does not mutate beyond drawing.
- `VisualizerModule` headless smoke test (`enabled=True, show_2d=False,
  show_3d=False`): `setup()` doesn't raise, doesn't spawn a subprocess; publishing
  `FrameEvent`/`PoseEvent`/`WorldPoseEvent` updates the cached latest-state.
- No test spawns the matplotlib subprocess or opens a cv2 window.

## Out of scope (YAGNI)

- Landmark undistortion before triangulation (tracked separately).
- Recording/playback of sessions.
- Multiple simultaneous hands.
- Camera exposure/gain control (tracked separately).
