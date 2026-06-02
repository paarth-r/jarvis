# Jarvis Hand Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-camera 2D hand-landmark overlays and an interactive 3D hand-skeleton viewer to the Jarvis runtime, replacing the placeholder DebugOverlayModule.

**Architecture:** A matplotlib 3D figure runs in a spawned subprocess (fed 21 points/frame over a drop-old `multiprocessing.Queue`), while a new `VisualizerModule` caches the latest frame/2D-pose/3D-pose per camera and, in a main-thread loop, draws cv2 2D overlays and pushes 3D points to the viewer. Pure helpers (`hand_points`, `draw_landmarks`, hand-connection constants) are unit-tested; GUI/subprocess code is never exercised in tests.

**Tech Stack:** Python 3.11+, asyncio, OpenCV (cv2 HighGUI), matplotlib (3D, `macosx` backend), multiprocessing (spawn), NumPy, pytest, pytest-asyncio

---

## File Map

```
jarvis/jarvis/modules/
  hand_landmarks.py     ← NEW: shared HAND_CONNECTIONS constant + pure helpers
  viewer_3d.py          ← NEW: Viewer3D subprocess wrapper (matplotlib)
  visualizer.py         ← NEW: VisualizerModule
  debug_overlay.py      ← DELETE (superseded)
jarvis/tests/
  test_hand_landmarks.py ← NEW: unit tests for pure helpers
  test_visualizer.py     ← NEW: headless smoke + caching tests
  test_debug_overlay.py  ← DELETE
jarvis/config.yaml       ← MODIFY: replace debug_overlay block with visualizer
```

---

## Task 1: Hand connections + pure helpers

**Files:**
- Create: `jarvis/modules/hand_landmarks.py`
- Create: `tests/test_hand_landmarks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hand_landmarks.py
import numpy as np
import pytest
from jarvis.modules.hand_landmarks import (
    HAND_CONNECTIONS, hand_points, draw_landmarks,
)


def test_hand_connections_are_valid_indices():
    # 21 landmarks → all endpoints in range, thumb+4 fingers+palm present
    assert len(HAND_CONNECTIONS) >= 20
    for a, b in HAND_CONNECTIONS:
        assert 0 <= a < 21 and 0 <= b < 21


def test_hand_points_maps_axes_x_z_negy():
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.1, 0.2, 0.5]   # world: x right, y down, z forward
    pts = hand_points(lm)
    assert len(pts) == 21
    # display = (x, z, -y)
    assert pts[0] == pytest.approx((0.1, 0.5, -0.2), abs=1e-6)


def test_hand_points_none_returns_21_nones():
    pts = hand_points(None)
    assert pts == [None] * 21


def test_draw_landmarks_mono_frame_returns_bgr():
    frame = np.zeros((800, 1280), dtype=np.uint8)  # mono
    lm2d = np.zeros((21, 3), dtype=np.float32)
    lm2d[:, 0] = 0.5
    lm2d[:, 1] = 0.5
    out = draw_landmarks(frame, lm2d, HAND_CONNECTIONS)
    assert out.shape == (800, 1280, 3)
    assert out.dtype == np.uint8


def test_draw_landmarks_does_not_mutate_input_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    before = frame.copy()
    lm2d = np.full((21, 3), 0.5, dtype=np.float32)
    draw_landmarks(frame, lm2d, HAND_CONNECTIONS)
    assert np.array_equal(frame, before)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hand_landmarks.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis.modules.hand_landmarks'`

- [ ] **Step 3: Write hand_landmarks.py**

```python
# jarvis/modules/hand_landmarks.py
from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe 21-landmark hand connections (thumb, index, middle, ring, pinky, palm)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]

Point3D = Optional[Tuple[float, float, float]]


def hand_points(world_landmarks: Optional[np.ndarray]) -> List[Point3D]:
    """Map (21,3) world landmarks (X right, Y down, Z fwd) to display tuples
    (right, depth, up) = (x, z, -y). None → list of 21 Nones."""
    if world_landmarks is None:
        return [None] * 21
    pts: List[Point3D] = []
    for i in range(world_landmarks.shape[0]):
        x, y, z = (float(v) for v in world_landmarks[i])
        pts.append((x, z, -y))
    return pts


def draw_landmarks(frame: np.ndarray, lm2d: np.ndarray, connections) -> np.ndarray:
    """Return a BGR copy of frame with normalized (21,3) landmarks drawn.
    Accepts mono (HxW) or BGR (HxWx3) input. Does not mutate the input."""
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        canvas = frame.copy()
    h, w = canvas.shape[:2]
    px = [(int(lm2d[i, 0] * w), int(lm2d[i, 1] * h)) for i in range(lm2d.shape[0])]
    for a, b in connections:
        if a < len(px) and b < len(px):
            cv2.line(canvas, px[a], px[b], (0, 255, 0), 2)
    for pt in px:
        cv2.circle(canvas, pt, 4, (0, 255, 255), -1)
    return canvas
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hand_landmarks.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add jarvis/modules/hand_landmarks.py tests/test_hand_landmarks.py
git commit -m "feat: add hand connections and pure 2D/3D landmark helpers"
```

---

## Task 2: Viewer3D subprocess wrapper

**Files:**
- Create: `jarvis/modules/viewer_3d.py`

No unit tests spawn the subprocess (GUI). This task only verifies the module
imports and that `hand_points` integration is sound (covered in Task 1). The
worker body is ported from Hyperform `tools/viewer_3d.py`, adapted to 21 hand
landmarks and the `(x, z, -y)` display mapping already applied by `hand_points`.

- [ ] **Step 1: Write viewer_3d.py**

```python
# jarvis/modules/viewer_3d.py
from __future__ import annotations

import multiprocessing
from typing import List, Optional, Tuple

from .hand_landmarks import HAND_CONNECTIONS

Point3D = Optional[Tuple[float, float, float]]

_NUM = 21
_BG = "#0d0d1a"
_BONE = "#00e676"
_JOINT = "#00bcd4"
_GRID = "#1e1e3a"
_LABEL = "#8888aa"
_DEFAULT_ELEV = 12.0
_DEFAULT_AZIM = -90.0


def _viewer_worker(queue: "multiprocessing.Queue") -> None:
    import matplotlib
    try:
        matplotlib.use("macosx")
    except Exception:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(7, 7), facecolor=_BG)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    try:
        fig.canvas.manager.set_window_title("Jarvis 3D Hand")
    except Exception:
        pass
    ax.view_init(elev=_DEFAULT_ELEV, azim=_DEFAULT_AZIM)
    plt.ion()
    plt.show(block=False)

    points: List[Point3D] = [None] * _NUM
    elev, azim = _DEFAULT_ELEV, _DEFAULT_AZIM

    while True:
        while True:
            try:
                item = queue.get_nowait()
                if item is None:
                    plt.close(fig)
                    return
                points = item
            except Exception:
                break
        try:
            elev, azim = ax.elev, ax.azim
        except Exception:
            pass
        _draw(ax, points, elev, azim, np)
        fig.canvas.draw_idle()
        plt.pause(0.033)


def _draw(ax, points: List[Point3D], elev: float, azim: float, np) -> None:
    ax.cla()
    ax.set_facecolor(_BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(_LABEL)
        axis.set_tick_params(colors=_LABEL, labelsize=7)
    ax.set_xlabel("X  → right")
    ax.set_ylabel("Y  → depth")
    ax.set_zlabel("Z  ↑ up")
    ax.view_init(elev=elev, azim=azim)

    xs = [p[0] if p else None for p in points]
    ys = [p[1] if p else None for p in points]
    zs = [p[2] if p else None for p in points]
    valid = [(xs[i], ys[i], zs[i]) for i in range(_NUM) if points[i] is not None]

    if not valid:
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(0.2, 0.8)
        ax.set_zlim(-0.2, 0.2)
        return

    vx, vy, vz = zip(*valid)
    span = max(max(vx) - min(vx), max(vy) - min(vy), max(vz) - min(vz), 0.2)
    half = span / 2.0 + 0.05
    cx, cy, cz = float(np.mean(vx)), float(np.mean(vy)), float(np.mean(vz))
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)

    for i, j in HAND_CONNECTIONS:
        if points[i] is not None and points[j] is not None:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                    color=_BONE, linewidth=2.0, solid_capstyle="round")
    ax.scatter([x for x in xs if x is not None],
               [y for y in ys if y is not None],
               [z for z in zs if z is not None],
               c=_JOINT, s=28, depthshade=False, zorder=5)


class Viewer3D:
    """Interactive 3D hand viewer in a spawned subprocess. Drag to rotate."""

    def __init__(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._queue: "multiprocessing.Queue" = ctx.Queue(maxsize=2)
        self._process = ctx.Process(
            target=_viewer_worker, args=(self._queue,),
            daemon=True, name="jarvis-3d-viewer",
        )

    def start(self) -> None:
        self._process.start()

    def update(self, points: List[Point3D]) -> None:
        try:
            self._queue.put_nowait(points)
        except Exception:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(points)
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
```

- [ ] **Step 2: Verify it imports**

Run: `python -c "from jarvis.modules.viewer_3d import Viewer3D; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add jarvis/modules/viewer_3d.py
git commit -m "feat: add Viewer3D matplotlib subprocess wrapper"
```

---

## Task 3: VisualizerModule

**Files:**
- Create: `jarvis/modules/visualizer.py`
- Create: `tests/test_visualizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_visualizer.py
import numpy as np
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.visualizer import VisualizerModule
from jarvis.events import FrameEvent, PoseEvent, WorldPoseEvent


def _make(show_2d=False, show_3d=False):
    return VisualizerModule("visualizer", {
        "enabled": True,
        "show_2d": show_2d,
        "show_3d": show_3d,
        "fps": 30,
        "cameras": {"left": {"width": 1280, "height": 800},
                    "right": {"width": 1280, "height": 800}},
    })


@pytest.mark.asyncio
async def test_headless_setup_does_not_spawn_or_raise():
    bus = EventBus()
    mod = _make(show_2d=False, show_3d=False)
    await mod.setup(bus)
    assert mod._viewer is None  # show_3d False → no subprocess


@pytest.mark.asyncio
async def test_caches_latest_events():
    bus = EventBus()
    mod = _make(show_2d=False, show_3d=False)
    await mod.setup(bus)

    frame = np.zeros((800, 1280), dtype=np.uint8)
    await bus.publish(FrameEvent(camera_id="left", frame=frame, timestamp=1.0))
    lm2d = np.full((21, 3), 0.5, dtype=np.float32)
    await bus.publish(PoseEvent(camera_id="left", landmarks_2d=lm2d, timestamp=1.0, hand_side="right"))
    lm3d = np.zeros((21, 3), dtype=np.float32)
    await bus.publish(WorldPoseEvent(landmarks_3d=lm3d, timestamp=1.0, confidence=1.0, hand_side="right"))

    assert mod._frames["left"] is not None
    assert mod._poses_2d["left"].shape == (21, 3)
    assert mod._world is not None
    assert mod._confidence == 1.0


@pytest.mark.asyncio
async def test_disabled_module_skips_subscriptions():
    bus = EventBus()
    mod = VisualizerModule("visualizer", {"enabled": False})
    await mod.setup(bus)
    # No crash; publishing does nothing harmful
    await bus.publish(WorldPoseEvent(landmarks_3d=np.zeros((21, 3), dtype=np.float32),
                                     timestamp=1.0, confidence=1.0, hand_side="right"))
    assert mod._world is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_visualizer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis.modules.visualizer'`

- [ ] **Step 3: Write visualizer.py**

```python
# jarvis/modules/visualizer.py
from __future__ import annotations
import asyncio
import logging
from typing import Dict, Optional

import numpy as np

from ..events import FrameEvent, PoseEvent, WorldPoseEvent
from ..runtime.module_base import Module
from .hand_landmarks import HAND_CONNECTIONS, hand_points, draw_landmarks

logger = logging.getLogger(__name__)


class VisualizerModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._enabled: bool = self.config.get("enabled", True)
        self._show_2d: bool = self.config.get("show_2d", True)
        self._show_3d: bool = self.config.get("show_3d", True)
        self._fps: int = self.config.get("fps", 30)
        self._cameras: Dict[str, dict] = self.config.get("cameras", {})

        self._frames: Dict[str, Optional[np.ndarray]] = {}
        self._poses_2d: Dict[str, Optional[np.ndarray]] = {}
        self._world: Optional[np.ndarray] = None
        self._confidence: float = 0.0
        self._viewer = None

        if not self._enabled:
            return

        if self._show_3d:
            from .viewer_3d import Viewer3D
            self._viewer = Viewer3D()
            self._viewer.start()

        self.bus.subscribe(FrameEvent, self._on_frame)
        self.bus.subscribe(PoseEvent, self._on_pose)
        self.bus.subscribe(WorldPoseEvent, self._on_world)

    async def _on_frame(self, event: FrameEvent) -> None:
        self._frames[event.camera_id] = event.frame

    async def _on_pose(self, event: PoseEvent) -> None:
        self._poses_2d[event.camera_id] = event.landmarks_2d

    async def _on_world(self, event: WorldPoseEvent) -> None:
        self._world = event.landmarks_3d
        self._confidence = event.confidence

    async def run(self) -> None:
        if not self._enabled or not (self._show_2d or self._show_3d):
            while True:
                await asyncio.sleep(3600)

        import cv2
        period = 1.0 / max(self._fps, 1)
        try:
            while True:
                if self._show_2d:
                    for cam_id, frame in list(self._frames.items()):
                        if frame is None:
                            continue
                        lm2d = self._poses_2d.get(cam_id)
                        if lm2d is not None:
                            img = draw_landmarks(frame, lm2d, HAND_CONNECTIONS)
                        elif frame.ndim == 2:
                            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        else:
                            img = frame
                        cv2.putText(img, f"{cam_id} conf:{self._confidence:.2f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2)
                        cv2.imshow(f"jarvis {cam_id}", img)
                if self._show_3d and self._viewer is not None:
                    self._viewer.update(hand_points(self._world))
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
                await asyncio.sleep(period)
        finally:
            cv2.destroyAllWindows()

    async def teardown(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_visualizer.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add jarvis/modules/visualizer.py tests/test_visualizer.py
git commit -m "feat: add VisualizerModule (2D overlay + 3D viewer driver)"
```

---

## Task 4: Remove DebugOverlayModule, wire config

**Files:**
- Delete: `jarvis/modules/debug_overlay.py`
- Delete: `tests/test_debug_overlay.py`
- Modify: `jarvis/config.yaml` (replace `debug_overlay` block)

- [ ] **Step 1: Delete the superseded module and its test**

```bash
git rm jarvis/modules/debug_overlay.py tests/test_debug_overlay.py
```

- [ ] **Step 2: Replace the config block**

In `jarvis/config.yaml`, replace:

```yaml
  debug_overlay:
    class: jarvis.modules.debug_overlay.DebugOverlayModule
    enabled: false
```

with:

```yaml
  visualizer:
    class: jarvis.modules.visualizer.VisualizerModule
    enabled: false        # opt-in; set true to show windows
    show_2d: true
    show_3d: true
    fps: 30
```

- [ ] **Step 3: Verify config loads and references resolve**

Run:
```bash
python -c "
from jarvis.runtime.config import load_config
c = load_config('config.yaml')
assert 'debug_overlay' not in c.modules
v = c.modules['visualizer']
assert v.class_path == 'jarvis.modules.visualizer.VisualizerModule'
import importlib
m, n = v.class_path.rsplit('.', 1)
getattr(importlib.import_module(m), n)
print('config ok')
"
```
Expected: `config ok`

- [ ] **Step 4: Run full suite**

Run: `python -m pytest tests/ -q`
Expected: all passed (no test_debug_overlay; new hand_landmarks + visualizer tests present)

- [ ] **Step 5: Commit**

```bash
git add jarvis/config.yaml
git commit -m "feat: replace debug_overlay with visualizer module in config"
```

---

## Task 5: Manual run verification (human-in-the-loop)

**Files:** none (operator step)

- [ ] **Step 1: Run the full runtime with the visualizer enabled**

Edit `config.yaml` → set `visualizer.enabled: true`. To verify *without*
mouse takeover, temporarily set `gesture_hci`/`action_dispatcher` `enabled:
false` (add `enabled: false` to those blocks), then:

```bash
cd /Users/paarth-r/Code/Jarvis/jarvis
python -m jarvis.main --config config.yaml
```

Expected: two `jarvis left` / `jarvis right` cv2 windows showing the camera
feeds with a green hand skeleton drawn when a hand is present, plus a "Jarvis 3D
Hand" matplotlib window with a rotatable 3D skeleton. ESC (on a cv2 window) or
Ctrl-C quits.

- [ ] **Step 2: Confirm clean shutdown**

Expected: Ctrl-C tears down all modules, closes windows, and the 3D subprocess
exits without a traceback.

---

## Self-Review Notes

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `Viewer3D` subprocess (matplotlib, queue, drag-rotate, equal aspect, dark) | Task 2 |
| 21-landmark hand skeleton + connections constant | Task 1 (constant), Task 2 (3D), Task 3 (2D) |
| Coord map (x,z,-y) | Task 1 (`hand_points`) |
| `VisualizerModule` caches frame/2D/3D per camera | Task 3 |
| Main-thread run loop: 2D cv2 + 3D push + ESC | Task 3 |
| show_2d / show_3d / fps config flags | Task 3, Task 4 |
| `hand_points`, `draw_landmarks` pure helpers unit-tested | Task 1 |
| Headless smoke + caching tests, no GUI/subprocess in tests | Task 3 |
| Remove DebugOverlayModule + test | Task 4 |
| Config block swap | Task 4 |

All spec requirements mapped. No placeholders. Types consistent
(`hand_points`/`draw_landmarks`/`HAND_CONNECTIONS` used identically across tasks).
