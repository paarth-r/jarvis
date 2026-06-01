# Jarvis Core Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Jarvis core runtime — async event bus, config-driven module system, dual OV9281 stereo pipeline, and gesture HCI — replacing the two-script prototype.

**Architecture:** Single asyncio event loop with typed pub/sub events as the communication spine. Camera and HandPose modules publish per-camera events; StereoFusionModule triangulates them into world-coordinate poses. Arm and LLM run as independent subprocesses connected via JSON-lines bridges with their own stack traces.

**Tech Stack:** Python 3.11+, asyncio, MediaPipe (mp.solutions.hands), OpenCV, NumPy, PyYAML, pyobjc-framework-Quartz (macOS mouse), pynput, pytest, pytest-asyncio

---

## File Map

```
jarvis/                          ← git repo root
  jarvis/                        ← Python package
    __init__.py
    events.py                    ← all event dataclasses
    main.py                      ← entry point
    runtime/
      __init__.py
      event_bus.py               ← EventBus: pub/sub over asyncio
      module_base.py             ← Module ABC
      config.py                  ← load_config(path) → JarvisConfig
      supervisor.py              ← Supervisor: lifecycle mgmt
      subprocess_bridge.py       ← SubprocessBridge: JSON-lines IPC
    modules/
      __init__.py
      camera.py                  ← CameraModule
      hand_pose.py               ← HandPoseModule
      stereo_fusion.py           ← StereoFusionModule
      gesture_hci.py             ← GestureHCIModule + PinchFSM
      action_dispatcher.py       ← ActionDispatcher
      debug_overlay.py           ← DebugOverlayModule
    processes/
      __init__.py
      arm_controller.py          ← stub subprocess
      llm_agent.py               ← stub subprocess
  tests/
    __init__.py
    test_event_bus.py
    test_config.py
    test_supervisor.py
    test_stereo_fusion.py
    test_subprocess_bridge.py
    test_gesture_hci.py
  config.yaml
  requirements.txt
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `jarvis/` package tree (all `__init__.py` files)
- Create: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directory tree and empty __init__.py files**

```bash
cd /Users/paarth-r/Code/Jarvis/jarvis
mkdir -p jarvis/runtime jarvis/modules jarvis/processes tests
touch jarvis/__init__.py
touch jarvis/runtime/__init__.py
touch jarvis/modules/__init__.py
touch jarvis/processes/__init__.py
touch tests/__init__.py
```

- [ ] **Step 2: Write requirements.txt**

```
opencv-python
mediapipe
numpy
pyyaml
pyobjc-framework-Quartz
pynput
pytest
pytest-asyncio
```

- [ ] **Step 3: Verify structure**

```bash
find jarvis tests -name "*.py" | sort
```

Expected output includes `jarvis/__init__.py`, `jarvis/runtime/__init__.py`, etc.

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 5: Commit**

```bash
git add jarvis/ tests/ requirements.txt
git commit -m "feat: scaffold jarvis package structure"
```

---

## Task 2: Event Dataclasses

**Files:**
- Create: `jarvis/events.py`
- Create: `tests/test_events.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_events.py
import numpy as np
import pytest
from jarvis.events import (
    FrameEvent, PoseEvent, WorldPoseEvent,
    IntentEvent, PointEvent, ActionEvent, SystemEvent,
)

def test_frame_event_fields():
    frame = np.zeros((800, 1280), dtype=np.uint8)
    e = FrameEvent(camera_id="left", frame=frame, timestamp=1.0)
    assert e.camera_id == "left"
    assert e.frame.shape == (800, 1280)
    assert e.timestamp == 1.0

def test_world_pose_event_fields():
    lm = np.zeros((21, 3), dtype=np.float32)
    e = WorldPoseEvent(landmarks_3d=lm, timestamp=2.0, confidence=1.0, hand_side="right")
    assert e.confidence == 1.0
    assert e.landmarks_3d.shape == (21, 3)

def test_intent_event_fields():
    e = IntentEvent(type="click", confidence=0.95, payload={"x": 0.5}, timestamp=3.0)
    assert e.type == "click"
    assert e.payload["x"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/paarth-r/Code/Jarvis/jarvis
python -m pytest tests/test_events.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.events'`

- [ ] **Step 3: Write events.py**

```python
# jarvis/events.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FrameEvent:
    camera_id: str
    frame: np.ndarray       # HxW uint8 mono (OV9281 native)
    timestamp: float        # time.perf_counter() seconds


@dataclass
class PoseEvent:
    camera_id: str
    landmarks_2d: np.ndarray  # (21, 3) normalized [x∈0-1, y∈0-1, z_rel]
    timestamp: float
    hand_side: str            # "left" | "right"


@dataclass
class WorldPoseEvent:
    # WorldPoseEvent is the stereo boundary.
    # Everything upstream is camera-aware; everything downstream is not.
    landmarks_3d: np.ndarray  # (21, 3) meters in world coordinate frame
    timestamp: float
    confidence: float         # 1.0 = full stereo, 0.5 = mono fallback
    hand_side: str


@dataclass
class IntentEvent:
    type: str        # "click" | "move" | "gesture" | "point" | "command"
    confidence: float
    payload: dict
    timestamp: float


@dataclass
class PointEvent:
    world_pos: np.ndarray   # ray origin in world coords (3,)
    direction: np.ndarray   # unit vector (3,)
    target: Optional[str]   # resolved target label, if known
    confidence: float
    timestamp: float


@dataclass
class ActionEvent:
    action_type: str
    params: dict
    timestamp: float


@dataclass
class SystemEvent:
    type: str        # "module_started" | "module_stopped" | "module_error"
    module_id: str
    payload: dict
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_events.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/events.py tests/test_events.py
git commit -m "feat: add typed event dataclasses"
```

---

## Task 3: EventBus

**Files:**
- Create: `jarvis/runtime/event_bus.py`
- Create: `tests/test_event_bus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_event_bus.py
import asyncio
import numpy as np
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.events import FrameEvent, SystemEvent


@pytest.mark.asyncio
async def test_subscribe_and_publish():
    bus = EventBus()
    received = []

    async def handler(event: FrameEvent):
        received.append(event)

    bus.subscribe(FrameEvent, handler)
    e = FrameEvent(camera_id="left", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(e)

    assert len(received) == 1
    assert received[0].camera_id == "left"


@pytest.mark.asyncio
async def test_wrong_type_not_delivered():
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe(FrameEvent, handler)
    await bus.publish(SystemEvent(type="test", module_id="x", payload={}))

    assert len(received) == 0


@pytest.mark.asyncio
async def test_multiple_subscribers_all_called():
    bus = EventBus()
    calls = [0, 0]

    async def h1(e): calls[0] += 1
    async def h2(e): calls[1] += 1

    bus.subscribe(FrameEvent, h1)
    bus.subscribe(FrameEvent, h2)

    e = FrameEvent(camera_id="left", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(e)

    assert calls == [1, 1]


@pytest.mark.asyncio
async def test_publish_with_no_subscribers_is_silent():
    bus = EventBus()
    # Should not raise
    await bus.publish(FrameEvent(camera_id="x", frame=np.zeros((10, 10), dtype=np.uint8), timestamp=0.0))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_event_bus.py -v
```

Expected: `ImportError: cannot import name 'EventBus'`

- [ ] **Step 3: Write event_bus.py**

```python
# jarvis/runtime/event_bus.py
from __future__ import annotations
from typing import Any, Awaitable, Callable, Dict, List, Type, TypeVar

T = TypeVar("T")


class EventBus:
    """Synchronous-dispatch pub/sub. publish() awaits all subscribers in sequence."""

    def __init__(self) -> None:
        self._subscribers: Dict[type, List[Callable[[Any], Awaitable[None]]]] = {}

    def subscribe(
        self,
        event_type: Type[T],
        callback: Callable[[T], Awaitable[None]],
    ) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    async def publish(self, event: Any) -> None:
        for callback in self._subscribers.get(type(event), []):
            await callback(event)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_event_bus.py -v
```

Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/runtime/event_bus.py tests/test_event_bus.py
git commit -m "feat: add EventBus pub/sub"
```

---

## Task 4: Module ABC + Config Loader

**Files:**
- Create: `jarvis/runtime/module_base.py`
- Create: `jarvis/runtime/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import tempfile
import pytest
from jarvis.runtime.config import load_config

FULL_CONFIG = """
cameras:
  left:
    device: 0
    width: 1280
    height: 800
    fps: 120
    intrinsics:
      fx: 910.0
      fy: 910.0
      cx: 640.0
      cy: 400.0
    extrinsics:
      R: [[1,0,0],[0,1,0],[0,0,1]]
      t: [-0.076, 0, 0]
  right:
    device: 1
    width: 1280
    height: 800
    fps: 120
    intrinsics:
      fx: 910.0
      fy: 910.0
      cx: 640.0
      cy: 400.0
    extrinsics:
      R: [[1,0,0],[0,1,0],[0,0,1]]
      t: [0.076, 0, 0]

modules:
  stereo_fusion:
    class: jarvis.modules.stereo_fusion.StereoFusionModule
    sync_tolerance_ms: 16.0
    mono_fallback: true
  debug_overlay:
    class: jarvis.modules.debug_overlay.DebugOverlayModule
    enabled: false

subprocesses:
  arm:
    enabled: false
    module: jarvis.processes.arm_controller
"""


def _write_tmp(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_cameras_loaded():
    path = _write_tmp(FULL_CONFIG)
    try:
        cfg = load_config(path)
        assert set(cfg.cameras.keys()) == {"left", "right"}
        left = cfg.cameras["left"]
        assert left.device == 0
        assert left.width == 1280
        assert left.fps == 120
        assert left.intrinsics["fx"] == 910.0
        assert left.extrinsics["t"] == [-0.076, 0, 0]
    finally:
        os.unlink(path)


def test_modules_loaded():
    path = _write_tmp(FULL_CONFIG)
    try:
        cfg = load_config(path)
        assert "stereo_fusion" in cfg.modules
        sf = cfg.modules["stereo_fusion"]
        assert sf.class_path == "jarvis.modules.stereo_fusion.StereoFusionModule"
        assert sf.enabled is True
        assert sf.settings["sync_tolerance_ms"] == 16.0

        assert cfg.modules["debug_overlay"].enabled is False
    finally:
        os.unlink(path)


def test_subprocesses_loaded():
    path = _write_tmp(FULL_CONFIG)
    try:
        cfg = load_config(path)
        assert "arm" in cfg.subprocesses
        assert cfg.subprocesses["arm"].enabled is False
        assert cfg.subprocesses["arm"].module == "jarvis.processes.arm_controller"
    finally:
        os.unlink(path)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/file.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_config.py -v
```

Expected: `ImportError: cannot import name 'load_config'`

- [ ] **Step 3: Write module_base.py**

```python
# jarvis/runtime/module_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .event_bus import EventBus


class Module(ABC):
    def __init__(self, module_id: str, config: dict) -> None:
        self.module_id = module_id
        self.config = config
        self.bus: Optional["EventBus"] = None

    async def setup(self, bus: "EventBus") -> None:
        """Wire subscriptions here. Call super().setup(bus) first."""
        self.bus = bus

    @abstractmethod
    async def run(self) -> None:
        """Main coroutine. Cancelled via asyncio.CancelledError on shutdown."""
        ...

    async def teardown(self) -> None:
        """Release hardware, close files, etc."""
```

- [ ] **Step 4: Write config.py**

```python
# jarvis/runtime/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import yaml


@dataclass
class CameraConfig:
    device: int
    width: int
    height: int
    fps: int
    intrinsics: Dict[str, float]
    extrinsics: Dict[str, Any]


@dataclass
class ModuleConfig:
    class_path: str
    enabled: bool
    settings: Dict[str, Any]


@dataclass
class SubprocessConfig:
    enabled: bool
    module: str


@dataclass
class JarvisConfig:
    cameras: Dict[str, CameraConfig]
    modules: Dict[str, ModuleConfig]
    subprocesses: Dict[str, SubprocessConfig]


def load_config(path: str) -> JarvisConfig:
    with open(path) as f:          # raises FileNotFoundError if missing
        raw = yaml.safe_load(f)

    cameras: Dict[str, CameraConfig] = {}
    for name, cam in (raw.get("cameras") or {}).items():
        cameras[name] = CameraConfig(
            device=cam["device"],
            width=cam.get("width", 1280),
            height=cam.get("height", 800),
            fps=cam.get("fps", 30),
            intrinsics=cam.get("intrinsics", {}),
            extrinsics=cam.get("extrinsics", {}),
        )

    modules: Dict[str, ModuleConfig] = {}
    for name, mod in (raw.get("modules") or {}).items():
        modules[name] = ModuleConfig(
            class_path=mod["class"],
            enabled=mod.get("enabled", True),
            settings={k: v for k, v in mod.items() if k not in ("class", "enabled")},
        )

    subprocesses: Dict[str, SubprocessConfig] = {}
    for name, sub in (raw.get("subprocesses") or {}).items():
        subprocesses[name] = SubprocessConfig(
            enabled=sub.get("enabled", False),
            module=sub.get("module", ""),
        )

    return JarvisConfig(cameras=cameras, modules=modules, subprocesses=subprocesses)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_config.py -v
```

Expected: 4 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add jarvis/runtime/module_base.py jarvis/runtime/config.py tests/test_config.py
git commit -m "feat: add Module ABC and config loader"
```

---

## Task 5: Supervisor

**Files:**
- Create: `jarvis/runtime/supervisor.py`
- Create: `tests/test_supervisor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_supervisor.py
import asyncio
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.runtime.module_base import Module
from jarvis.runtime.supervisor import Supervisor
from jarvis.runtime.config import JarvisConfig, CameraConfig, ModuleConfig, SubprocessConfig


def _empty_config() -> JarvisConfig:
    return JarvisConfig(cameras={}, modules={}, subprocesses={})


class _LifecycleModule(Module):
    def __init__(self, module_id, config):
        super().__init__(module_id, config)
        self.setup_called = False
        self.run_called = False
        self.teardown_called = False

    async def setup(self, bus):
        await super().setup(bus)
        self.setup_called = True

    async def run(self):
        self.run_called = True
        # Immediately exit so gather() completes
        return

    async def teardown(self):
        self.teardown_called = True


@pytest.mark.asyncio
async def test_supervisor_calls_lifecycle():
    cfg = _empty_config()
    sup = Supervisor(cfg)

    mod = _LifecycleModule("test_mod", {})
    sup._inject_module(mod)

    await sup.start()

    assert mod.setup_called
    assert mod.run_called
    assert mod.teardown_called


@pytest.mark.asyncio
async def test_supervisor_bus_is_shared():
    cfg = _empty_config()
    sup = Supervisor(cfg)

    buses = []

    class _BusCaptureModule(Module):
        async def setup(self, bus):
            await super().setup(bus)
            buses.append(bus)
        async def run(self):
            return

    sup._inject_module(_BusCaptureModule("a", {}))
    sup._inject_module(_BusCaptureModule("b", {}))
    await sup.start()

    assert len(buses) == 2
    assert buses[0] is buses[1]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_supervisor.py -v
```

Expected: `ImportError: cannot import name 'Supervisor'`

- [ ] **Step 3: Write supervisor.py**

```python
# jarvis/runtime/supervisor.py
from __future__ import annotations
import asyncio
import importlib
import logging
from typing import List

from .event_bus import EventBus
from .module_base import Module
from .config import JarvisConfig
from .subprocess_bridge import SubprocessBridge
from ..events import SystemEvent

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(self, config: JarvisConfig) -> None:
        self._config = config
        self._bus = EventBus()
        self._modules: List[Module] = []
        self._bridges: List[SubprocessBridge] = []

    # ── internal helper used by tests ──────────────────────────────────────
    def _inject_module(self, module: Module) -> None:
        self._modules.append(module)

    # ── public ─────────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._build_modules()

        for mod in self._modules:
            await mod.setup(self._bus)

        for name, sub_cfg in self._config.subprocesses.items():
            if sub_cfg.enabled:
                bridge = SubprocessBridge(name, sub_cfg.module, self._bus)
                self._bridges.append(bridge)

        tasks = [asyncio.create_task(mod.run()) for mod in self._modules]
        tasks += [asyncio.create_task(b.run()) for b in self._bridges]

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except (asyncio.CancelledError, Exception) as exc:
            if not isinstance(exc, asyncio.CancelledError):
                logger.exception("Module raised an exception")
                await self._bus.publish(SystemEvent(
                    type="module_error",
                    module_id="supervisor",
                    payload={"error": str(exc)},
                ))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for mod in reversed(self._modules):
                try:
                    await mod.teardown()
                except Exception:
                    logger.exception("teardown error in %s", mod.module_id)

    @property
    def bus(self) -> EventBus:
        return self._bus

    # ── private ────────────────────────────────────────────────────────────
    def _build_modules(self) -> None:
        from ..modules.camera import CameraModule
        from ..modules.hand_pose import HandPoseModule

        # One CameraModule + HandPoseModule per declared camera
        cameras_dict = {
            cam_id: {
                "camera_id": cam_id,
                "device": cam.device,
                "width": cam.width,
                "height": cam.height,
                "fps": cam.fps,
                "intrinsics": cam.intrinsics,
                "extrinsics": cam.extrinsics,
            }
            for cam_id, cam in self._config.cameras.items()
        }

        for cam_id, cam in self._config.cameras.items():
            self._modules.append(CameraModule(f"camera_{cam_id}", cameras_dict[cam_id]))
            self._modules.append(HandPoseModule(f"hand_pose_{cam_id}", {"camera_id": cam_id}))

        # Declared modules — inject cameras into every module's config
        for mod_id, mod_cfg in self._config.modules.items():
            if not mod_cfg.enabled:
                continue
            cls = self._load_class(mod_cfg.class_path)
            full_cfg = {**mod_cfg.settings, "cameras": cameras_dict}
            self._modules.append(cls(mod_id, full_cfg))

    @staticmethod
    def _load_class(class_path: str):
        module_path, class_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_supervisor.py -v
```

Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/runtime/supervisor.py tests/test_supervisor.py
git commit -m "feat: add Supervisor lifecycle management"
```

---

## Task 6: SubprocessBridge

**Files:**
- Create: `jarvis/runtime/subprocess_bridge.py`
- Create: `tests/test_subprocess_bridge.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subprocess_bridge.py
import asyncio
import sys
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.runtime.subprocess_bridge import SubprocessBridge


# A fake subprocess that emits one JSON event then exits cleanly
FAKE_PROC_SCRIPT = """
import sys, json, time
print(json.dumps({"type": "ping", "payload": {"msg": "hello"}, "timestamp": 1.0}), flush=True)
# Wait for shutdown command
for line in sys.stdin:
    data = json.loads(line)
    if data.get("cmd") == "shutdown":
        break
sys.exit(0)
"""


@pytest.mark.asyncio
async def test_bridge_restarts_on_nonzero_exit():
    bus = EventBus()
    start_count = [0]
    original_run_once = SubprocessBridge._run_once

    async def counting_run_once(self):
        start_count[0] += 1
        if start_count[0] < 3:
            raise RuntimeError("simulated crash")
        # On 3rd attempt, cancel to stop the loop
        raise asyncio.CancelledError()

    SubprocessBridge._run_once = counting_run_once
    bridge = SubprocessBridge("test", "fake.module", bus)
    bridge._backoff = 0.001  # speed up test

    try:
        await asyncio.wait_for(bridge.run(), timeout=1.0)
    except asyncio.CancelledError:
        pass
    finally:
        SubprocessBridge._run_once = original_run_once

    assert start_count[0] == 3


@pytest.mark.asyncio
async def test_bridge_send_writes_json():
    bus = EventBus()
    bridge = SubprocessBridge("test", "fake.module", bus)

    written = []

    class _FakeStdin:
        async def drain(self): pass
        def write(self, data): written.append(data)

    class _FakeProc:
        stdin = _FakeStdin()

    bridge._proc = _FakeProc()
    await bridge.send("shutdown", {"reason": "test"})

    import json
    assert len(written) == 1
    msg = json.loads(written[0].decode())
    assert msg["cmd"] == "shutdown"
    assert msg["params"]["reason"] == "test"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_subprocess_bridge.py -v
```

Expected: `ImportError: cannot import name 'SubprocessBridge'`

- [ ] **Step 3: Write subprocess_bridge.py**

```python
# jarvis/runtime/subprocess_bridge.py
from __future__ import annotations
import asyncio
import json
import logging
import sys
from typing import Optional

from .event_bus import EventBus

logger = logging.getLogger(__name__)


class SubprocessBridge:
    """
    Spawns a child process, exchanges JSON-lines over stdin/stdout.
    stderr is inherited so child stack traces appear in the parent terminal.
    Restarts the child on non-zero exit with exponential backoff.
    """

    def __init__(self, bridge_id: str, module_path: str, bus: EventBus) -> None:
        self._bridge_id = bridge_id
        self._module_path = module_path
        self._bus = bus
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._backoff = 1.0

    async def run(self) -> None:
        while True:
            try:
                await self._run_once()
                # clean exit — don't restart
                logger.info("Bridge %s exited cleanly", self._bridge_id)
                return
            except asyncio.CancelledError:
                await self._shutdown()
                raise
            except Exception as exc:
                logger.error("Bridge %s crashed: %s", self._bridge_id, exc)

            logger.info("Bridge %s restarting in %.1fs", self._bridge_id, self._backoff)
            await asyncio.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, 30.0)

    async def _run_once(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", self._module_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,   # inherit — child stack traces appear in parent terminal
        )
        self._backoff = 1.0

        async for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                await self._dispatch(data)
            except json.JSONDecodeError:
                logger.warning("Bridge %s bad JSON: %r", self._bridge_id, line)

        await self._proc.wait()
        if self._proc.returncode != 0:
            raise RuntimeError(f"Process exited with code {self._proc.returncode}")

    async def _dispatch(self, data: dict) -> None:
        # Extended in future tasks when processes emit real events
        pass

    async def send(self, cmd: str, params: dict | None = None) -> None:
        if self._proc and self._proc.stdin:
            msg = json.dumps({"cmd": cmd, "params": params or {}}) + "\n"
            self._proc.stdin.write(msg.encode())
            await self._proc.stdin.drain()

    async def _shutdown(self) -> None:
        if self._proc:
            try:
                await self.send("shutdown")
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                self._proc.kill()
            self._proc = None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_subprocess_bridge.py -v
```

Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/runtime/subprocess_bridge.py tests/test_subprocess_bridge.py
git commit -m "feat: add SubprocessBridge with restart backoff"
```

---

## Task 7: CameraModule

**Files:**
- Create: `jarvis/modules/camera.py`

(CameraModule wraps cv2.VideoCapture blocking I/O; the test stubs VideoCapture to avoid hardware.)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_camera.py
import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.camera import CameraModule
from jarvis.events import FrameEvent


def _make_module(camera_id="left", device=0):
    return CameraModule(f"camera_{camera_id}", {
        "camera_id": camera_id,
        "device": device,
        "width": 1280,
        "height": 800,
        "fps": 30,
    })


@pytest.mark.asyncio
async def test_camera_publishes_frame_event():
    bus = EventBus()
    received = []

    async def on_frame(e: FrameEvent):
        received.append(e)

    bus.subscribe(FrameEvent, on_frame)

    fake_frame = np.zeros((800, 1280), dtype=np.uint8)

    call_count = [0]

    def fake_read():
        call_count[0] += 1
        if call_count[0] > 2:
            raise asyncio.CancelledError()
        return True, fake_frame

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = fake_read

    mod = _make_module()
    await mod.setup(bus)
    mod._cap = mock_cap

    with pytest.raises((asyncio.CancelledError, Exception)):
        await asyncio.wait_for(mod.run(), timeout=1.0)

    assert len(received) >= 1
    assert received[0].camera_id == "left"
    assert received[0].frame.shape == (800, 1280)


@pytest.mark.asyncio
async def test_camera_raises_if_device_not_open():
    bus = EventBus()
    mod = _make_module()
    await mod.setup(bus)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mod._cap = mock_cap

    with pytest.raises(RuntimeError, match="failed to open"):
        await mod.run()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_camera.py -v
```

Expected: `ImportError: cannot import name 'CameraModule'`

- [ ] **Step 3: Write camera.py**

```python
# jarvis/modules/camera.py
from __future__ import annotations
import asyncio
import time
from typing import Optional

import cv2
import numpy as np

from ..events import FrameEvent
from ..runtime.module_base import Module


class CameraModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._camera_id: str = self.config["camera_id"]
        self._device: int = self.config["device"]
        self._width: int = self.config.get("width", 1280)
        self._height: int = self.config.get("height", 800)
        self._fps: int = self.config.get("fps", 30)
        self._cap: Optional[cv2.VideoCapture] = None

    async def run(self) -> None:
        loop = asyncio.get_event_loop()
        if self._cap is None:
            self._cap = await loop.run_in_executor(None, self._open_capture)
        if not self._cap.isOpened():
            raise RuntimeError(f"Camera device {self._device} failed to open")

        try:
            while True:
                frame = await loop.run_in_executor(None, self._read_frame)
                if frame is None:
                    await asyncio.sleep(0.001)
                    continue
                await self.bus.publish(FrameEvent(
                    camera_id=self._camera_id,
                    frame=frame,
                    timestamp=time.perf_counter(),
                ))
        finally:
            if self._cap:
                self._cap.release()
                self._cap = None

    async def teardown(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        return cap

    def _read_frame(self) -> Optional[np.ndarray]:
        if not self._cap:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_camera.py -v
```

Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/modules/camera.py tests/test_camera.py
git commit -m "feat: add CameraModule"
```

---

## Task 8: HandPoseModule

**Files:**
- Create: `jarvis/modules/hand_pose.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hand_pose.py
import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.hand_pose import HandPoseModule
from jarvis.events import FrameEvent, PoseEvent


def _make_pose_module(camera_id="left"):
    return HandPoseModule(f"hand_pose_{camera_id}", {"camera_id": camera_id})


@pytest.mark.asyncio
async def test_wrong_camera_id_ignored():
    bus = EventBus()
    received = []

    async def on_pose(e: PoseEvent):
        received.append(e)

    bus.subscribe(PoseEvent, on_pose)

    mod = _make_pose_module(camera_id="left")
    await mod.setup(bus)

    # Publish a frame from "right" — should be ignored
    frame = FrameEvent(camera_id="right", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(frame)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_no_hand_detected_publishes_nothing():
    bus = EventBus()
    received = []

    async def on_pose(e: PoseEvent):
        received.append(e)

    bus.subscribe(PoseEvent, on_pose)

    mod = _make_pose_module(camera_id="left")

    # Patch _process to return None (no hand found)
    mod._process = lambda event: None
    await mod.setup(bus)

    frame = FrameEvent(camera_id="left", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(frame)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_hand_detected_publishes_pose_event():
    bus = EventBus()
    received = []

    async def on_pose(e: PoseEvent):
        received.append(e)

    bus.subscribe(PoseEvent, on_pose)

    mod = _make_pose_module(camera_id="left")

    fake_landmarks = np.zeros((21, 3), dtype=np.float32)
    fake_pose = PoseEvent(
        camera_id="left",
        landmarks_2d=fake_landmarks,
        timestamp=1.0,
        hand_side="right",
    )
    mod._process = lambda event: fake_pose
    await mod.setup(bus)

    frame = FrameEvent(camera_id="left", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(frame)

    assert len(received) == 1
    assert received[0].camera_id == "left"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_hand_pose.py -v
```

Expected: `ImportError: cannot import name 'HandPoseModule'`

- [ ] **Step 3: Write hand_pose.py**

```python
# jarvis/modules/hand_pose.py
from __future__ import annotations
import asyncio
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from ..events import FrameEvent, PoseEvent
from ..runtime.module_base import Module


class HandPoseModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._camera_id: str = self.config["camera_id"]
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.bus.subscribe(FrameEvent, self._on_frame)

    async def _on_frame(self, event: FrameEvent) -> None:
        if event.camera_id != self._camera_id:
            return
        loop = asyncio.get_event_loop()
        pose_event = await loop.run_in_executor(None, self._process, event)
        if pose_event is not None:
            await self.bus.publish(pose_event)

    def _process(self, event: FrameEvent) -> Optional[PoseEvent]:
        # OV9281 outputs mono; convert to RGB for MediaPipe
        if event.frame.ndim == 2:
            rgb = cv2.cvtColor(event.frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(event.frame, cv2.COLOR_BGR2RGB)

        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0]
        hand_side = "right"
        if result.multi_handedness:
            hand_side = result.multi_handedness[0].classification[0].label.lower()

        landmarks = np.array(
            [[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32
        )
        return PoseEvent(
            camera_id=self._camera_id,
            landmarks_2d=landmarks,
            timestamp=event.timestamp,
            hand_side=hand_side,
        )

    async def run(self) -> None:
        # Subscriptions handle all work; keep coroutine alive
        while True:
            await asyncio.sleep(3600)

    async def teardown(self) -> None:
        self._hands.close()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_hand_pose.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/modules/hand_pose.py tests/test_hand_pose.py
git commit -m "feat: add HandPoseModule"
```

---

## Task 9: StereoFusionModule

**Files:**
- Create: `jarvis/modules/stereo_fusion.py`
- Create: `tests/test_stereo_fusion.py`

The extrinsic convention in config.yaml is **camera-to-world** (t = camera position in world). `_build_projection_matrix` converts to world-to-camera for `cv2.triangulatePoints`: `R_wc = R_c2w.T`, `t_wc = -R_c2w.T @ t_c2w`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stereo_fusion.py
import asyncio
import numpy as np
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.stereo_fusion import StereoFusionModule
from jarvis.events import PoseEvent, WorldPoseEvent

CAMERAS = {
    "left": {
        "device": 0, "width": 1280, "height": 800,
        "intrinsics": {"fx": 910.0, "fy": 910.0, "cx": 640.0, "cy": 400.0},
        "extrinsics": {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [-0.076, 0, 0]},
    },
    "right": {
        "device": 1, "width": 1280, "height": 800,
        "intrinsics": {"fx": 910.0, "fy": 910.0, "cx": 640.0, "cy": 400.0},
        "extrinsics": {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.076, 0, 0]},
    },
}


def _project_world_to_norm(world_pt, cam_cfg):
    """Project a 3D world point to normalized [0-1] image coords."""
    R_c2w = np.array(cam_cfg["extrinsics"]["R"], dtype=np.float64)
    t_c2w = np.array(cam_cfg["extrinsics"]["t"], dtype=np.float64)
    intr = cam_cfg["intrinsics"]
    w, h = cam_cfg["width"], cam_cfg["height"]

    R_wc = R_c2w.T
    t_wc = -R_c2w.T @ t_c2w
    p_cam = R_wc @ np.array(world_pt, dtype=np.float64) + t_wc

    x_px = intr["fx"] * p_cam[0] / p_cam[2] + intr["cx"]
    y_px = intr["fy"] * p_cam[1] / p_cam[2] + intr["cy"]

    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 0] = x_px / w
    lm[:, 1] = y_px / h
    return lm


async def _make_module(bus):
    mod = StereoFusionModule("stereo_fusion", {
        "sync_tolerance_ms": 16.0,
        "mono_fallback": True,
        "cameras": CAMERAS,
    })
    await mod.setup(bus)
    return mod


@pytest.mark.asyncio
async def test_full_stereo_triangulates_correctly():
    bus = EventBus()
    received = []

    async def on_world(e: WorldPoseEvent):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    await _make_module(bus)

    world_pt = [0.0, 0.0, 0.5]

    # Publish secondary (right) first so it's buffered when primary (left) arrives
    await bus.publish(PoseEvent(
        camera_id="right",
        landmarks_2d=_project_world_to_norm(world_pt, CAMERAS["right"]),
        timestamp=1.0,
        hand_side="right",
    ))
    await bus.publish(PoseEvent(
        camera_id="left",
        landmarks_2d=_project_world_to_norm(world_pt, CAMERAS["left"]),
        timestamp=1.005,
        hand_side="right",
    ))

    assert len(received) == 1
    assert received[0].confidence == 1.0
    # All 21 landmarks were projected from the same world point; wrist (0) should round-trip
    assert np.allclose(received[0].landmarks_3d[0], world_pt, atol=0.02)


@pytest.mark.asyncio
async def test_mono_fallback_when_one_camera_missing():
    bus = EventBus()
    received = []

    async def on_world(e: WorldPoseEvent):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    await _make_module(bus)

    lm = _project_world_to_norm([0.0, 0.0, 0.5], CAMERAS["left"])
    await bus.publish(PoseEvent(
        camera_id="left", landmarks_2d=lm, timestamp=2.0, hand_side="right",
    ))
    # No right camera pose — should mono-fallback
    await asyncio.sleep(0.02)  # wait past sync tolerance

    assert len(received) == 1
    assert received[0].confidence == 0.5


@pytest.mark.asyncio
async def test_out_of_tolerance_poses_not_fused():
    bus = EventBus()
    received = []

    async def on_world(e: WorldPoseEvent):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    await _make_module(bus)

    world_pt = [0.0, 0.0, 0.5]

    await bus.publish(PoseEvent(
        camera_id="right",
        landmarks_2d=_project_world_to_norm(world_pt, CAMERAS["right"]),
        timestamp=1.0,
        hand_side="right",
    ))
    # Left arrives 100ms later — outside 16ms tolerance
    await bus.publish(PoseEvent(
        camera_id="left",
        landmarks_2d=_project_world_to_norm(world_pt, CAMERAS["left"]),
        timestamp=1.100,
        hand_side="right",
    ))

    # Fusion should not have happened (left only sees itself → mono fallback)
    assert all(e.confidence < 1.0 for e in received)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_stereo_fusion.py -v
```

Expected: `ImportError: cannot import name 'StereoFusionModule'`

- [ ] **Step 3: Write stereo_fusion.py**

```python
# jarvis/modules/stereo_fusion.py
from __future__ import annotations
import asyncio
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..events import PoseEvent, WorldPoseEvent
from ..runtime.module_base import Module


class StereoFusionModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._tolerance: float = self.config.get("sync_tolerance_ms", 16.0) / 1000.0
        self._mono_fallback: bool = self.config.get("mono_fallback", True)
        self._mono_depth: float = self.config.get("mono_depth_m", 0.5)
        self._camera_configs: Dict[str, dict] = self.config.get("cameras", {})

        camera_ids = list(self._camera_configs.keys())
        self._primary: str = camera_ids[0] if camera_ids else ""
        self._buffers: Dict[str, deque] = {cid: deque(maxlen=20) for cid in camera_ids}
        self._last_fused_ts: float = -float("inf")

        self.bus.subscribe(PoseEvent, self._on_pose)

    async def _on_pose(self, event: PoseEvent) -> None:
        if event.camera_id not in self._buffers:
            return
        self._buffers[event.camera_id].append(event)

        # Only the primary camera triggers fusion to avoid duplicates
        if event.camera_id != self._primary:
            return

        world_event = self._try_fuse(event.timestamp)
        if world_event is not None:
            self._last_fused_ts = world_event.timestamp
            await self.bus.publish(world_event)

    def _try_fuse(self, ref_ts: float) -> Optional[WorldPoseEvent]:
        camera_ids = list(self._camera_configs.keys())
        matched: Dict[str, PoseEvent] = {}
        for cid in camera_ids:
            best = self._closest(cid, ref_ts)
            if best is not None:
                matched[cid] = best

        if len(matched) == len(camera_ids):
            return self._triangulate(matched)

        if self._mono_fallback and matched:
            cid = next(iter(matched))
            return self._mono_project(matched[cid])

        return None

    def _closest(self, cam_id: str, ref_ts: float) -> Optional[PoseEvent]:
        best: Optional[PoseEvent] = None
        best_dt = float("inf")
        for event in self._buffers.get(cam_id, []):
            dt = abs(event.timestamp - ref_ts)
            if dt < best_dt and dt <= self._tolerance:
                best_dt = dt
                best = event
        return best

    def _build_proj(self, cam_cfg: dict) -> np.ndarray:
        intr = cam_cfg["intrinsics"]
        K = np.array([
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # Config stores camera-to-world; convert to world-to-camera for cv2
        R_c2w = np.array(cam_cfg["extrinsics"]["R"], dtype=np.float64)
        t_c2w = np.array(cam_cfg["extrinsics"]["t"], dtype=np.float64)
        R_wc = R_c2w.T
        t_wc = (-R_c2w.T @ t_c2w).reshape(3, 1)
        return K @ np.hstack([R_wc, t_wc])

    def _triangulate(self, matched: Dict[str, PoseEvent]) -> WorldPoseEvent:
        ids = list(matched.keys())
        cam_a, cam_b = ids[0], ids[1]
        cfg_a, cfg_b = self._camera_configs[cam_a], self._camera_configs[cam_b]

        P1 = self._build_proj(cfg_a)
        P2 = self._build_proj(cfg_b)

        def to_px(lm2d: np.ndarray, cfg: dict) -> np.ndarray:
            return (lm2d[:, :2] * np.array([cfg["width"], cfg["height"]])).T.astype(np.float64)

        pts_a = to_px(matched[cam_a].landmarks_2d, cfg_a)
        pts_b = to_px(matched[cam_b].landmarks_2d, cfg_b)

        pts4d = cv2.triangulatePoints(P1, P2, pts_a, pts_b)
        pts3d = (pts4d[:3] / pts4d[3]).T.astype(np.float32)

        avg_ts = (matched[cam_a].timestamp + matched[cam_b].timestamp) / 2.0
        return WorldPoseEvent(
            landmarks_3d=pts3d,
            timestamp=avg_ts,
            confidence=1.0,
            hand_side=matched[cam_a].hand_side,
        )

    def _mono_project(self, event: PoseEvent) -> WorldPoseEvent:
        cfg = self._camera_configs[event.camera_id]
        intr = cfg["intrinsics"]
        w, h = cfg["width"], cfg["height"]
        d = self._mono_depth

        lm_px = event.landmarks_2d[:, :2] * np.array([w, h])
        lm3d = np.zeros((21, 3), dtype=np.float32)
        lm3d[:, 0] = (lm_px[:, 0] - intr["cx"]) * d / intr["fx"]
        lm3d[:, 1] = (lm_px[:, 1] - intr["cy"]) * d / intr["fy"]
        lm3d[:, 2] = d

        return WorldPoseEvent(
            landmarks_3d=lm3d,
            timestamp=event.timestamp,
            confidence=0.5,
            hand_side=event.hand_side,
        )

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_stereo_fusion.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/modules/stereo_fusion.py tests/test_stereo_fusion.py
git commit -m "feat: add StereoFusionModule with triangulation and mono fallback"
```

---

## Task 10: GestureHCIModule

**Files:**
- Create: `jarvis/modules/gesture_hci.py`
- Create: `tests/test_gesture_hci.py`

Ported from `mouse_control/gestures.py` and `mouse_control/control.py`. The `PinchFSM` is extracted as a testable pure class (no hardware). `GestureHCIModule` wraps it with `MouseController` from the existing `mouse_control/control.py`.

MediaPipe landmark indices used: `INDEX_MCP=5`, `THUMB_MCP=2`, `INDEX_TIP=8`, `THUMB_TIP=4`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gesture_hci.py
import pytest
from jarvis.modules.gesture_hci import PinchFSM


def test_idle_when_far():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    assert fsm.update(0.10) == "idle"
    assert fsm.update(0.08) == "idle"


def test_click_on_pinch():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    assert fsm.update(0.10) == "idle"
    assert fsm.update(0.01) == "click"   # transition: open → held


def test_no_double_click_while_held():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    fsm.update(0.10)
    fsm.update(0.01)          # click
    assert fsm.update(0.01) == "idle"   # still held — no second click
    assert fsm.update(0.01) == "idle"


def test_click_again_after_release():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    fsm.update(0.01)  # click (held)
    fsm.update(0.10)  # release
    assert fsm.update(0.01) == "click"  # new click


def test_hysteresis_prevents_bounce():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    fsm.update(0.01)   # held
    # Distance between thresh and release_thresh — should stay held
    assert fsm.update(0.03) == "idle"   # 0.02 < 0.03 < 0.04, still held
    assert fsm.update(0.05) == "idle"   # above release_thresh — now open
    assert fsm.update(0.01) == "click"  # new pinch
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_gesture_hci.py -v
```

Expected: `ImportError: cannot import name 'PinchFSM'`

- [ ] **Step 3: Write gesture_hci.py**

```python
# jarvis/modules/gesture_hci.py
from __future__ import annotations
import asyncio
from enum import Enum
from typing import Optional

import numpy as np

from ..events import WorldPoseEvent, IntentEvent
from ..runtime.module_base import Module

# MediaPipe hand landmark indices
INDEX_MCP = 5
THUMB_MCP = 2
INDEX_TIP = 8
THUMB_TIP = 4


class _PinchState(Enum):
    OPEN = "open"
    HELD = "held"


class PinchFSM:
    """Pure FSM — no hardware dependency. Returns 'click' on transition OPEN→HELD."""

    def __init__(
        self,
        pinch_thresh: float = 0.025,   # meters; enter held
        release_thresh: float = 0.045,  # meters; exit held (hysteresis)
    ) -> None:
        self._thresh = pinch_thresh
        self._release = release_thresh
        self._state = _PinchState.OPEN

    def update(self, distance: float) -> str:
        if self._state == _PinchState.OPEN:
            if distance < self._thresh:
                self._state = _PinchState.HELD
                return "click"
            return "idle"

        if self._state == _PinchState.HELD:
            if distance > self._release:
                self._state = _PinchState.OPEN
            return "idle"

        return "idle"


class GestureHCIModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._fsm = PinchFSM(
            pinch_thresh=self.config.get("pinch_thresh_m", 0.025),
            release_thresh=self.config.get("release_thresh_m", 0.045),
        )
        # Import MouseController lazily so Quartz import failure doesn't break tests
        from mouse_control.control import MouseController
        self._controller = MouseController(
            smoothing=self.config.get("smoothing", 0.3),
            move_threshold_px=self.config.get("move_threshold_px", 2.0),
        )
        self._gain: float = self.config.get("screen_gain", 2.5)
        self.bus.subscribe(WorldPoseEvent, self._on_pose)

    async def _on_pose(self, event: WorldPoseEvent) -> None:
        lm = event.landmarks_3d

        # Cursor follows midpoint of index and thumb knuckles (MCPs), mirrored on X
        mid_x = (lm[INDEX_MCP, 0] + lm[THUMB_MCP, 0]) / 2.0
        mid_y = (lm[INDEX_MCP, 1] + lm[THUMB_MCP, 1]) / 2.0
        norm_x = max(0.0, min(1.0, 1.0 - (0.5 + self._gain * mid_x)))
        norm_y = max(0.0, min(1.0, 0.5 + self._gain * mid_y))

        self._controller.move(norm_x, norm_y)

        dist = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]))
        intent = self._fsm.update(dist)

        if intent == "click":
            self._controller.click(norm_x, norm_y)

        await self.bus.publish(IntentEvent(
            type=intent,
            confidence=event.confidence,
            payload={"norm_x": norm_x, "norm_y": norm_y, "dist_m": dist},
            timestamp=event.timestamp,
        ))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_gesture_hci.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/modules/gesture_hci.py tests/test_gesture_hci.py
git commit -m "feat: add GestureHCIModule with PinchFSM"
```

---

## Task 11: ActionDispatcher

**Files:**
- Create: `jarvis/modules/action_dispatcher.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_action_dispatcher.py
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.action_dispatcher import ActionDispatcher
from jarvis.events import ActionEvent
import time


@pytest.mark.asyncio
async def test_unknown_action_does_not_raise():
    bus = EventBus()
    mod = ActionDispatcher("action_dispatcher", {})
    await mod.setup(bus)

    # Should log a warning, not raise
    await bus.publish(ActionEvent(action_type="nonexistent", params={}, timestamp=time.perf_counter()))


@pytest.mark.asyncio
async def test_raycast_action_calls_hotkey():
    bus = EventBus()
    mod = ActionDispatcher("action_dispatcher", {})
    await mod.setup(bus)

    with patch("jarvis.modules.action_dispatcher.pyautogui") as mock_pyautogui:
        await bus.publish(ActionEvent(
            action_type="raycast", params={}, timestamp=time.perf_counter()
        ))
        mock_pyautogui.hotkey.assert_called_once_with("command", "space")


@pytest.mark.asyncio
async def test_cooldown_prevents_rapid_repeat():
    bus = EventBus()
    mod = ActionDispatcher("action_dispatcher", {"cooldown_s": 1.0})
    await mod.setup(bus)

    call_count = [0]

    with patch("jarvis.modules.action_dispatcher.pyautogui") as mock_pyautogui:
        mock_pyautogui.hotkey.side_effect = lambda *a: call_count.__setitem__(0, call_count[0] + 1)
        ts = time.perf_counter()
        await bus.publish(ActionEvent(action_type="raycast", params={}, timestamp=ts))
        await bus.publish(ActionEvent(action_type="raycast", params={}, timestamp=ts + 0.1))

    assert call_count[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_action_dispatcher.py -v
```

Expected: `ImportError: cannot import name 'ActionDispatcher'`

- [ ] **Step 3: Write action_dispatcher.py**

```python
# jarvis/modules/action_dispatcher.py
from __future__ import annotations
import asyncio
import logging
import time
from typing import Callable, Dict

import pyautogui
from pynput.keyboard import Controller, Key

from ..events import ActionEvent
from ..runtime.module_base import Module

logger = logging.getLogger(__name__)
_keyboard = Controller()


def _raycast() -> None:
    pyautogui.hotkey("command", "space")


def _media_next() -> None:
    _keyboard.press(Key.media_next)
    _keyboard.release(Key.media_next)


def _media_prev() -> None:
    _keyboard.press(Key.media_previous)
    _keyboard.release(Key.media_previous)


def _media_play_pause() -> None:
    _keyboard.press(Key.media_play_pause)
    _keyboard.release(Key.media_play_pause)


def _volume_up() -> None:
    _keyboard.press(Key.media_volume_up)
    _keyboard.release(Key.media_volume_up)


HANDLERS: Dict[str, Callable[[], None]] = {
    "raycast": _raycast,
    "media_next": _media_next,
    "media_prev": _media_prev,
    "media_play_pause": _media_play_pause,
    "volume_up": _volume_up,
}


class ActionDispatcher(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._cooldown: float = self.config.get("cooldown_s", 1.0)
        self._last_ts: float = -float("inf")
        self.bus.subscribe(ActionEvent, self._on_action)

    async def _on_action(self, event: ActionEvent) -> None:
        if time.perf_counter() - self._last_ts < self._cooldown:
            return
        handler = HANDLERS.get(event.action_type)
        if handler is None:
            logger.warning("Unknown action: %s", event.action_type)
            return
        handler()
        self._last_ts = time.perf_counter()

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_action_dispatcher.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/modules/action_dispatcher.py tests/test_action_dispatcher.py
git commit -m "feat: add ActionDispatcher with cooldown"
```

---

## Task 12: DebugOverlayModule

**Files:**
- Create: `jarvis/modules/debug_overlay.py`

No unit tests (OpenCV visual output). Verify it imports and initialises cleanly.

- [ ] **Step 1: Write a smoke import test**

```python
# tests/test_debug_overlay.py
import pytest
from jarvis.modules.debug_overlay import DebugOverlayModule
from jarvis.runtime.event_bus import EventBus


@pytest.mark.asyncio
async def test_debug_overlay_sets_up_without_error():
    bus = EventBus()
    mod = DebugOverlayModule("debug_overlay", {"enabled": False})
    # setup should not raise even when OpenCV window won't open
    await mod.setup(bus)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_debug_overlay.py -v
```

Expected: `ImportError: cannot import name 'DebugOverlayModule'`

- [ ] **Step 3: Write debug_overlay.py**

```python
# jarvis/modules/debug_overlay.py
from __future__ import annotations
import asyncio
import time
from typing import Optional

import cv2
import numpy as np

from ..events import FrameEvent, WorldPoseEvent, IntentEvent, SystemEvent
from ..runtime.module_base import Module

# MediaPipe hand connections (landmark index pairs)
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]


class DebugOverlayModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._enabled: bool = self.config.get("enabled", True)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_pose: Optional[WorldPoseEvent] = None
        self._latest_intent: Optional[IntentEvent] = None
        self._fps: float = 0.0
        self._last_frame_ts: float = time.perf_counter()

        if self._enabled:
            self.bus.subscribe(FrameEvent, self._on_frame)
            self.bus.subscribe(WorldPoseEvent, self._on_pose)
            self.bus.subscribe(IntentEvent, self._on_intent)

    async def _on_frame(self, event: FrameEvent) -> None:
        now = time.perf_counter()
        dt = now - self._last_frame_ts
        self._fps = 0.1 * (1.0 / max(dt, 1e-6)) + 0.9 * self._fps
        self._last_frame_ts = now
        self._latest_frame = event.frame.copy()

    async def _on_pose(self, event: WorldPoseEvent) -> None:
        self._latest_pose = event

    async def _on_intent(self, event: IntentEvent) -> None:
        self._latest_intent = event

    async def run(self) -> None:
        if not self._enabled:
            while True:
                await asyncio.sleep(3600)
            return

        loop = asyncio.get_event_loop()
        while True:
            await loop.run_in_executor(None, self._render)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            await asyncio.sleep(0)

        cv2.destroyAllWindows()

    def _render(self) -> None:
        if self._latest_frame is None:
            return

        if self._latest_frame.ndim == 2:
            display = cv2.cvtColor(self._latest_frame, cv2.COLOR_GRAY2BGR)
        else:
            display = self._latest_frame.copy()
        display = cv2.flip(display, 1)
        h, w = display.shape[:2]

        pose = self._latest_pose
        if pose is not None:
            lm = pose.landmarks_3d
            # Back-project to rough pixel coords using first camera (approx)
            lm_px = []
            for i in range(21):
                # Simple depth-normalized projection back to display
                px = int((1.0 - (0.5 + lm[i, 0])) * w)
                py = int((0.5 + lm[i, 1]) * h)
                lm_px.append((px, py))
            for a, b in _CONNECTIONS:
                if 0 <= a < len(lm_px) and 0 <= b < len(lm_px):
                    cv2.line(display, lm_px[a], lm_px[b], (0, 255, 0), 2)
            for pt in lm_px:
                cv2.circle(display, pt, 4, (0, 255, 255), -1)

            conf_str = f"conf:{pose.confidence:.2f}"
            cv2.putText(display, conf_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        intent = self._latest_intent
        if intent is not None:
            cv2.putText(
                display, f"Intent: {intent.type}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        cv2.putText(
            display, f"FPS: {self._fps:.1f}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1,
        )
        cv2.imshow("Jarvis Debug", display)

    async def teardown(self) -> None:
        if self._enabled:
            cv2.destroyAllWindows()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_debug_overlay.py -v
```

Expected: 1 test PASSED

- [ ] **Step 5: Commit**

```bash
git add jarvis/modules/debug_overlay.py tests/test_debug_overlay.py
git commit -m "feat: add DebugOverlayModule"
```

---

## Task 13: Process Stubs (arm + LLM)

**Files:**
- Create: `jarvis/processes/arm_controller.py`
- Create: `jarvis/processes/llm_agent.py`

Both stubs read JSON-lines from stdin and exit cleanly on `{"cmd":"shutdown"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_process_stubs.py
import asyncio
import json
import sys
import pytest


async def _run_stub(module: str, timeout: float = 2.0) -> int:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", module,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    msg = json.dumps({"cmd": "shutdown", "params": {}}) + "\n"
    proc.stdin.write(msg.encode())
    await proc.stdin.drain()
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    return proc.returncode


@pytest.mark.asyncio
async def test_arm_controller_shuts_down_cleanly():
    code = await _run_stub("jarvis.processes.arm_controller")
    assert code == 0


@pytest.mark.asyncio
async def test_llm_agent_shuts_down_cleanly():
    code = await _run_stub("jarvis.processes.llm_agent")
    assert code == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_process_stubs.py -v
```

Expected: `ModuleNotFoundError: No module named 'jarvis.processes.arm_controller'` (or similar)

- [ ] **Step 3: Write arm_controller.py**

```python
# jarvis/processes/arm_controller.py
"""
6-DOF arm controller subprocess.
Reads JSON-lines commands from stdin; writes JSON-lines status to stdout.
This is a stub — arm hardware integration is a future sub-project.
"""
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[arm] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("arm_controller started (stub)")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("bad JSON: %r", line)
            continue

        cmd = msg.get("cmd")
        if cmd == "shutdown":
            logger.info("arm_controller shutting down")
            break
        else:
            logger.info("arm_controller stub — ignoring cmd: %s", cmd)

    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write llm_agent.py**

```python
# jarvis/processes/llm_agent.py
"""
LLM agent subprocess.
Reads JSON-lines IntentEvents from stdin; writes JSON-lines ActionEvents to stdout.
This is a stub — Claude API integration is a future sub-project.
"""
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[llm] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("llm_agent started (stub)")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("bad JSON: %r", line)
            continue

        cmd = msg.get("cmd")
        if cmd == "shutdown":
            logger.info("llm_agent shutting down")
            break
        else:
            logger.info("llm_agent stub — ignoring: %s", cmd)

    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_process_stubs.py -v
```

Expected: 2 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add jarvis/processes/arm_controller.py jarvis/processes/llm_agent.py tests/test_process_stubs.py
git commit -m "feat: add arm and llm process stubs"
```

---

## Task 14: main.py + config.yaml

**Files:**
- Create: `jarvis/main.py`
- Create: `config.yaml`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main.py
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from jarvis.main import build_supervisor


def test_build_supervisor_returns_supervisor(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
cameras:
  left:
    device: 0
    width: 1280
    height: 800
    fps: 120
    intrinsics: {fx: 910.0, fy: 910.0, cx: 640.0, cy: 400.0}
    extrinsics:
      R: [[1,0,0],[0,1,0],[0,0,1]]
      t: [-0.076, 0, 0]

modules:
  debug_overlay:
    class: jarvis.modules.debug_overlay.DebugOverlayModule
    enabled: false

subprocesses:
  arm:
    enabled: false
    module: jarvis.processes.arm_controller
""")
    from jarvis.runtime.supervisor import Supervisor
    sup = build_supervisor(str(cfg_file))
    assert isinstance(sup, Supervisor)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_main.py -v
```

Expected: `ImportError: cannot import name 'build_supervisor'`

- [ ] **Step 3: Write main.py**

```python
# jarvis/main.py
from __future__ import annotations
import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from .runtime.config import load_config
from .runtime.supervisor import Supervisor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_supervisor(config_path: str) -> Supervisor:
    cfg = load_config(config_path)
    return Supervisor(cfg)


async def _run(config_path: str) -> None:
    sup = build_supervisor(config_path)
    loop = asyncio.get_event_loop()

    def _handle_signal():
        logger.info("Shutdown signal received")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    try:
        await sup.start()
    except asyncio.CancelledError:
        logger.info("Jarvis stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis runtime")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run(str(config_path)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write config.yaml**

```yaml
# Jarvis runtime configuration

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
    extrinsics:           # camera position/orientation in world frame (camera-to-world)
      R: [[1,0,0],[0,1,0],[0,0,1]]
      t: [-0.076, 0, 0]   # meters; left camera at -3 inches on X axis
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
      t: [0.076, 0, 0]    # meters; right camera at +3 inches on X axis

modules:
  stereo_fusion:
    class: jarvis.modules.stereo_fusion.StereoFusionModule
    sync_tolerance_ms: 16.0
    mono_fallback: true
    mono_depth_m: 0.5
  gesture_hci:
    class: jarvis.modules.gesture_hci.GestureHCIModule
    smoothing: 0.3
    move_threshold_px: 2.0
    screen_gain: 2.5
    pinch_thresh_m: 0.025
    release_thresh_m: 0.045
  action_dispatcher:
    class: jarvis.modules.action_dispatcher.ActionDispatcher
    cooldown_s: 1.0
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

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_main.py -v
```

Expected: 1 test PASSED

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASSED (any test requiring real hardware is skipped or uses mocks)

- [ ] **Step 7: Commit**

```bash
git add jarvis/main.py config.yaml tests/test_main.py
git commit -m "feat: add main entry point and default config.yaml"
```

---

## Self-Review Notes

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Async event bus | Task 3 |
| Module ABC | Task 4 |
| Config YAML | Task 4 |
| Supervisor lifecycle | Task 5 |
| SubprocessBridge + restart | Task 6 |
| CameraModule per-camera | Task 7 |
| HandPoseModule per-camera | Task 8 |
| StereoFusionModule + mono fallback | Task 9 |
| GestureHCIModule + PinchFSM | Task 10 |
| ActionDispatcher | Task 11 |
| DebugOverlayModule | Task 12 |
| Subprocess stubs (arm + llm) | Task 13 |
| main.py entry point | Task 14 |
| cameras injected into module config | Task 5 (`_build_modules`) |
| WorldPoseEvent as stereo boundary | Tasks 9–10 |
| config.yaml structure | Task 4 + Task 14 |

All spec requirements covered.
