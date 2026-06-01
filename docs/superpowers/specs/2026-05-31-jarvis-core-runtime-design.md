# Jarvis Core Runtime — Design Spec

**Date:** 2026-05-31  
**Status:** Approved  
**Scope:** Core runtime only (sub-project 1 of 5)

---

## Roadmap Context

The full Jarvis system will be built in layers:

1. **Core runtime** ← this spec
2. Spatial pointing (room + screen awareness)
3. Mouse + HCI (existing code may survive as-is)
4. Voice + LLM layer
5. DUMMY arm (6-DOF)
6. Projector integration (room as interactive screen)

Each layer's spec is written separately. This spec covers only the core runtime.

---

## Goals

- Replace the two-script prototype (`mouse_control/`, `gesture_control/`) with a clean, extensible runtime
- Provide a stable spine that each future subsystem plugs into identically
- Keep the camera/gesture tight loop fast; isolate heavy subsystems (arm, LLM) as separate processes with their own stack traces
- Adding a new module or camera should require config changes only, not code changes

---

## Architecture

### Communication Model: Hybrid Async Event Bus

- **Main process**: single `asyncio` event loop. All latency-sensitive modules (camera, hand pose, stereo fusion, gesture HCI) run as async coroutines on the main loop. CPU-bound work runs via `loop.run_in_executor`.
- **Heavy subprocesses**: arm controller and LLM agent each run as separate OS processes, connected to the main process via `SubprocessBridge` (JSON-lines over stdin/stdout). Each has its own stack trace and crash-restart cycle.
- **No inter-module imports**: modules only import from `events.py` and `runtime/`. All data exchange goes through the `EventBus`.

### Visibility

Headless by default. Optional `DebugOverlayModule` (OpenCV window) toggled via config. No persistent on-screen UI.

### Module Registration

Python plugin classes declared in `config.yaml`. The `Supervisor` instantiates modules by dotted class path at startup. Enabling/disabling a module is a one-line config change.

---

## Project Structure

```
jarvis/
  runtime/
    event_bus.py          # EventBus: subscribe/publish, asyncio queues
    module_base.py        # Module ABC: setup / run / teardown
    supervisor.py         # load modules from config, manage lifecycle
    subprocess_bridge.py  # spawn/restart child processes, JSON-lines protocol
    config.py             # load + validate config.yaml
  modules/
    camera.py             # CameraModule: capture → FrameEvent
    hand_pose.py          # HandPoseModule: FrameEvent → PoseEvent
    stereo_fusion.py      # StereoFusionModule: N×PoseEvent → WorldPoseEvent
    gesture_hci.py        # GestureHCIModule: WorldPoseEvent → IntentEvent + mouse
    debug_overlay.py      # DebugOverlayModule: renders skeleton/state/FPS
    action_dispatcher.py  # ActionDispatcher: IntentEvent → system actions
  processes/
    arm_controller.py     # standalone: reads JSON cmds, controls 6-DOF arm
    llm_agent.py          # standalone: reads IntentEvents, calls Claude API
  events.py               # all event dataclasses
  main.py                 # entry point
config.yaml               # module declarations + camera configs + settings
```

---

## Event Types (`events.py`)

All events are plain dataclasses treated as immutable by convention. `timestamp` is `time.perf_counter()` seconds throughout. (`frozen=True` is avoided because `np.ndarray` fields make `__hash__` fail at runtime if the event is ever hashed.)

```python
@dataclass
class FrameEvent:
    camera_id: str
    frame: np.ndarray       # HxW uint8 mono (OV9281 native)
    timestamp: float

@dataclass
class PoseEvent:
    camera_id: str
    landmarks_2d: np.ndarray  # (21, 3) normalized [x, y, z]
    timestamp: float
    hand_side: str            # "left" | "right"

@dataclass
class WorldPoseEvent:
    landmarks_3d: np.ndarray  # (21, 3) meters, world coordinate frame
    timestamp: float
    confidence: float         # 1.0 = full stereo, <1.0 = mono fallback
    hand_side: str

@dataclass
class IntentEvent:
    type: str                 # "click" | "move" | "gesture" | "point" | "command"
    confidence: float
    payload: dict
    timestamp: float

@dataclass
class PointEvent:
    world_pos: np.ndarray     # ray origin in world coords
    direction: np.ndarray     # unit vector
    target: Optional[str]     # identified target, if resolved
    confidence: float
    timestamp: float

@dataclass
class ActionEvent:
    action_type: str
    params: dict
    timestamp: float

@dataclass
class SystemEvent:
    type: str                 # "module_started" | "module_stopped" | "module_error"
    module_id: str
    payload: dict
```

`WorldPoseEvent` is the stereo boundary. Everything above it is camera-aware; everything below it operates purely in world coordinates (meters, relative to room origin). No downstream module knows how many cameras exist.

---

## Module Interface (`runtime/module_base.py`)

```python
class Module(ABC):
    def __init__(self, module_id: str, config: dict) -> None:
        self.module_id = module_id
        self.config = config
        self.bus: Optional[EventBus] = None

    async def setup(self, bus: EventBus) -> None:
        self.bus = bus
        # subclasses wire subscriptions here

    @abstractmethod
    async def run(self) -> None:
        # main loop; cancelled via asyncio.CancelledError on shutdown
        ...

    async def teardown(self) -> None:
        # release hardware, close files, etc.
        pass
```

---

## Supervisor (`runtime/supervisor.py`)

1. Parse `config.yaml` via `config.py`
2. Instantiate each enabled module by dotted class path (e.g. `jarvis.modules.camera.CameraModule`)
3. For each camera declared in `config.cameras`, create one `CameraModule` and one `HandPoseModule` instance, injecting the camera's config
4. Call `module.setup(bus)` on all modules
5. Run all `module.run()` coroutines concurrently via `asyncio.gather(..., return_exceptions=True)`
6. On module exception: publish `SystemEvent(type="module_error")`, log full traceback, optionally restart with backoff
7. On SIGINT/SIGTERM: cancel all tasks, call `teardown()` on each module in reverse order
8. Spawn `SubprocessBridge` for each enabled entry in `config.subprocesses`

---

## Camera Pipeline

### `CameraModule`

- One instance per camera declared in `config.cameras`
- Opens `cv2.VideoCapture(device)` in `loop.run_in_executor` (blocking I/O)
- Sets resolution and fps on the capture object
- Publishes `FrameEvent(camera_id, frame, timestamp)` at configured fps
- OV9281 outputs mono uint8; frame is passed through without color conversion

### `HandPoseModule`

- One instance per camera (same `camera_id`)
- Subscribes to `FrameEvent` filtered by its `camera_id`
- Runs MediaPipe `HandLandmarker` in executor (CPU-bound)
- MediaPipe accepts mono frames natively (converts to grayscale internally anyway)
- Publishes `PoseEvent(camera_id, landmarks_2d, timestamp, hand_side)`
- Drops frame silently if MediaPipe returns no hand

### `StereoFusionModule`

- Subscribes to all `PoseEvent`s regardless of `camera_id`
- Maintains a per-camera ring buffer (last N poses by timestamp)
- On each new `PoseEvent`: attempt to find a matching pose from every other configured camera within `sync_tolerance_ms` (default 16 ms)
- **Full stereo**: all cameras have a match → triangulate via `cv2.triangulatePoints` using per-camera intrinsics + extrinsics → publish `WorldPoseEvent(confidence=1.0)`
- **Mono fallback** (if `mono_fallback: true`): only one camera sees a hand → project onto a plane at estimated depth (0.5 m default) → publish `WorldPoseEvent(confidence=0.5)`
- Extrinsics are loaded from config; no calibration logic in this module

**Adding a camera**: add one block to `config.cameras`. `StereoFusionModule` discovers the camera list from config at startup; no code changes.

---

## Downstream Modules

### `GestureHCIModule`

- Subscribes to `WorldPoseEvent`
- Ports existing `GestureFSM` and `MouseController` logic from `mouse_control/`
- Maps 3D hand position to screen normalized coords
- Uses Quartz (macOS) for mouse injection
- Publishes `IntentEvent(type="click"|"move")` in addition to injecting mouse events directly

### `ActionDispatcher`

- Subscribes to `ActionEvent`
- Routes to handlers: keyboard/media keys (pynput), shell commands, Raycast, etc.
- Ports existing `ACTIONS` dict from `gesture_control/utils.py`

### `DebugOverlayModule`

- `enabled: false` by default in config
- Subscribes to `FrameEvent`, `WorldPoseEvent`, `IntentEvent`, `SystemEvent`
- Renders: hand skeleton (projected back to 2D), 3D world position, gesture state, FPS, latency, per-module status
- Single OpenCV window; press Esc or close window to stop

---

## Subprocess Bridge (`runtime/subprocess_bridge.py`)

```
Main process                    Child process
─────────────                   ─────────────
SubprocessBridge
  stdin writer  ──JSON-lines──▶  reads commands
  stdout reader ◀─JSON-lines──  writes events/status
  stderr        ◀─passthrough─  logs (for stack traces)
```

- Spawns child via `asyncio.create_subprocess_exec`
- Command format: `{"cmd": "...", "params": {...}, "id": "..."}`
- Event format: `{"type": "...", "payload": {...}, "timestamp": ...}`
- On child exit with non-zero code: log stderr, restart after backoff (1s → 2s → 4s → max 30s)
- Shutdown sequence: send `{"cmd": "shutdown"}`, wait up to 3 s, then `SIGKILL`
- Bridge deserializes child events and re-publishes onto the main `EventBus`

Both `processes/arm_controller.py` and `processes/llm_agent.py` start as stubs that handle the shutdown command and log a "not yet implemented" message. They are `enabled: false` in `config.yaml`.

---

## `config.yaml` Structure

```yaml
cameras:
  left:
    device: 0
    width: 1280
    height: 800
    fps: 120
    intrinsics:           # placeholder — replace with calibrated values
      fx: 910.0
      fy: 910.0
      cx: 640.0
      cy: 400.0
    extrinsics:
      R: [[1,0,0],[0,1,0],[0,0,1]]
      t: [-0.076, 0, 0]   # meters; -3 inches
  right:
    device: 1
    width: 1280
    height: 800
    fps: 120
    intrinsics:           # placeholder — replace with calibrated values
      fx: 910.0
      fy: 910.0
      cx: 640.0
      cy: 400.0
    extrinsics:
      R: [[1,0,0],[0,1,0],[0,0,1]]
      t: [0.076, 0, 0]    # meters; +3 inches

modules:
  stereo_fusion:
    class: jarvis.modules.stereo_fusion.StereoFusionModule
    sync_tolerance_ms: 16.0
    mono_fallback: true
  gesture_hci:
    class: jarvis.modules.gesture_hci.GestureHCIModule
    smoothing: 0.3
    move_threshold_px: 2.0
  action_dispatcher:
    class: jarvis.modules.action_dispatcher.ActionDispatcher
  debug_overlay:
    class: jarvis.modules.debug_overlay.DebugOverlayModule
    enabled: false

subprocesses:
  arm:
    enabled: false
    module: jarvis.processes.arm_controller
  llm:
    enabled: false
    module: jarvis.processes.llm_agent
```

---

## What Is Out of Scope

- Camera calibration utility (already exists externally; config accepts calibrated values)
- Spatial pointing logic (sub-project 2)
- Voice / LLM integration (sub-project 4)
- Arm control implementation (sub-project 5)
- Projector integration (sub-project 6)
- Windows support (macOS only; Quartz dependency)

---

## Dependencies

```
opencv-python
mediapipe
numpy
pyobjc-framework-Quartz   # mouse injection (macOS)
pynput                     # keyboard/media actions
pyyaml                     # config loading
```

No new dependencies beyond what the existing codebase already uses.
