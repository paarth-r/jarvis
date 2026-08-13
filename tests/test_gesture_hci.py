import numpy as np
import pytest

from jarvis.events import WorldPoseEvent, IntentEvent
from jarvis.modules.gesture_hci import PinchFSM, GestureHCIModule, THUMB_TIP, INDEX_TIP
from jarvis.runtime.event_bus import EventBus


def _fsm():
    return PinchFSM(pinch_thresh=0.02, release_thresh=0.04, hold_delay=0.2)


def test_idle_when_far():
    fsm = _fsm()
    assert fsm.update(0.10, 1.0) == "idle"
    assert fsm.update(0.08, 1.1) == "idle"


def test_quick_pinch_is_click():
    fsm = _fsm()
    assert fsm.update(0.10, 1.0) == "idle"
    assert fsm.update(0.01, 1.1) == "idle"     # pinch closes — pending, nothing yet
    assert fsm.update(0.10, 1.2) == "click"    # opens within hold delay — click


def test_held_pinch_becomes_drag():
    fsm = _fsm()
    fsm.update(0.01, 1.0)                       # pinch closes — pending
    assert fsm.update(0.01, 1.1) == "idle"      # 0.1s — still pending
    assert fsm.update(0.01, 1.25) == "press"    # >0.2s held — drag starts
    assert fsm.update(0.01, 1.4) == "idle"      # still dragging, no repeat
    assert fsm.update(0.10, 1.5) == "release"   # opens — drag ends


def test_no_click_after_drag_release():
    fsm = _fsm()
    fsm.update(0.01, 1.0)
    fsm.update(0.01, 1.25)   # press
    assert fsm.update(0.10, 1.4) == "release"
    assert fsm.update(0.10, 1.5) == "idle"


def test_pinch_again_after_click():
    fsm = _fsm()
    fsm.update(0.01, 1.0)
    fsm.update(0.10, 1.1)    # click
    fsm.update(0.01, 1.2)    # pending again
    assert fsm.update(0.01, 1.45) == "press"   # held — drag


def test_hysteresis_prevents_bounce_while_dragging():
    fsm = _fsm()
    fsm.update(0.01, 1.0)
    fsm.update(0.01, 1.25)                      # press (dragging)
    assert fsm.update(0.03, 1.3) == "idle"      # 0.02 < 0.03 < 0.04 — still held
    assert fsm.update(0.05, 1.4) == "release"   # above release_thresh
    fsm.update(0.01, 1.5)
    assert fsm.update(0.10, 1.6) == "click"     # quick pinch again — click


def test_hysteresis_applies_while_pending():
    fsm = _fsm()
    fsm.update(0.01, 1.0)                       # pending
    assert fsm.update(0.03, 1.1) == "idle"      # between thresholds — still pending
    assert fsm.update(0.03, 1.25) == "press"    # held past delay — drag


# --- GestureHCIModule: click vs pinch-hold drag ---

class FakeController:
    def __init__(self, **kwargs):
        self.calls = []

    def move(self, x, y):
        self.calls.append(("move", x, y))

    def mouse_down(self, x, y):
        self.calls.append(("mouse_down", x, y))

    def mouse_up(self, x, y):
        self.calls.append(("mouse_up", x, y))

    def click(self, x, y):
        self.calls.append(("click", x, y))


def _pose(pinch_dist: float, ts: float) -> WorldPoseEvent:
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 2] = 0.5
    lm[INDEX_TIP, 0] = pinch_dist  # thumb tip stays at x=0
    return WorldPoseEvent(landmarks_3d=lm, timestamp=ts, confidence=1.0, hand_side="right")


async def _make_module(bus, monkeypatch):
    controller = FakeController()
    import mouse_control.control
    monkeypatch.setattr(mouse_control.control, "MouseController", lambda **kw: controller)
    mod = GestureHCIModule("gesture_hci", {
        "pinch_thresh_m": 0.02,
        "release_thresh_m": 0.04,
        "drag_hold_s": 0.2,
    })
    await mod.setup(bus)
    return controller


@pytest.mark.asyncio
async def test_quick_pinch_dispatches_single_click(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch)

    await bus.publish(_pose(0.10, 1.0))
    await bus.publish(_pose(0.01, 1.05))  # closes
    await bus.publish(_pose(0.10, 1.15))  # opens 0.1s later — click

    kinds = [c[0] for c in controller.calls]
    assert kinds.count("click") == 1
    assert "mouse_down" not in kinds
    assert "mouse_up" not in kinds


@pytest.mark.asyncio
async def test_held_pinch_dispatches_down_move_up(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch)

    await bus.publish(_pose(0.10, 1.0))
    await bus.publish(_pose(0.01, 1.1))   # closes — pending
    await bus.publish(_pose(0.01, 1.35))  # held past 0.2s — button down
    await bus.publish(_pose(0.01, 1.5))   # still held — drag movement
    await bus.publish(_pose(0.10, 1.6))   # opens — button up

    kinds = [c[0] for c in controller.calls]
    assert "mouse_down" in kinds
    assert "mouse_up" in kinds
    assert kinds.index("mouse_down") < kinds.index("mouse_up")
    assert "click" not in kinds
    down_i, up_i = kinds.index("mouse_down"), kinds.index("mouse_up")
    assert "move" in kinds[down_i:up_i]


@pytest.mark.asyncio
async def test_intents_published_for_click_and_drag(monkeypatch):
    bus = EventBus()
    received = []

    async def on_intent(e: IntentEvent):
        received.append(e.type)

    bus.subscribe(IntentEvent, on_intent)
    await _make_module(bus, monkeypatch)

    await bus.publish(_pose(0.01, 1.0))   # pending
    await bus.publish(_pose(0.10, 1.1))   # click
    await bus.publish(_pose(0.01, 1.2))   # pending
    await bus.publish(_pose(0.01, 1.45))  # press
    await bus.publish(_pose(0.10, 1.6))   # release

    assert received == ["idle", "click", "idle", "press", "release"]
