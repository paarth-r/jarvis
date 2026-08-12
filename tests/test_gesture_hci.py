import numpy as np
import pytest

from jarvis.events import WorldPoseEvent, IntentEvent
from jarvis.modules.gesture_hci import PinchFSM, GestureHCIModule, THUMB_TIP, INDEX_TIP
from jarvis.runtime.event_bus import EventBus


def test_idle_when_far():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    assert fsm.update(0.10) == "idle"
    assert fsm.update(0.08) == "idle"


def test_press_on_pinch():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    assert fsm.update(0.10) == "idle"
    assert fsm.update(0.01) == "press"   # transition: open → held


def test_no_repeat_press_while_held():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    fsm.update(0.10)
    fsm.update(0.01)          # press
    assert fsm.update(0.01) == "idle"   # still held — no second press
    assert fsm.update(0.01) == "idle"


def test_release_on_unpinch():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    fsm.update(0.01)  # press (held)
    assert fsm.update(0.10) == "release"   # transition: held → open


def test_press_again_after_release():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    fsm.update(0.01)  # press (held)
    fsm.update(0.10)  # release
    assert fsm.update(0.01) == "press"  # new press


def test_hysteresis_prevents_bounce():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04)
    fsm.update(0.01)   # held
    # Distance between thresh and release_thresh — should stay held
    assert fsm.update(0.03) == "idle"      # 0.02 < 0.03 < 0.04, still held
    assert fsm.update(0.05) == "release"   # above release_thresh — now open
    assert fsm.update(0.01) == "press"     # new pinch


# --- GestureHCIModule: pinch-hold drives mouse_down / drag / mouse_up ---

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
    })
    await mod.setup(bus)
    return controller


@pytest.mark.asyncio
async def test_pinch_hold_drag_release_dispatches_down_move_up(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch)

    await bus.publish(_pose(0.10, 1.0))   # open hand — just moves
    await bus.publish(_pose(0.01, 1.1))   # pinch closes — button down
    await bus.publish(_pose(0.01, 1.2))   # held — keeps moving (drag)
    await bus.publish(_pose(0.10, 1.3))   # pinch opens — button up

    kinds = [c[0] for c in controller.calls]
    assert "mouse_down" in kinds
    assert "mouse_up" in kinds
    assert kinds.index("mouse_down") < kinds.index("mouse_up")
    assert "click" not in kinds
    # cursor still tracked while held (drag)
    down_i, up_i = kinds.index("mouse_down"), kinds.index("mouse_up")
    assert "move" in kinds[down_i:up_i]


@pytest.mark.asyncio
async def test_press_and_release_intents_published(monkeypatch):
    bus = EventBus()
    received = []

    async def on_intent(e: IntentEvent):
        received.append(e.type)

    bus.subscribe(IntentEvent, on_intent)
    await _make_module(bus, monkeypatch)

    await bus.publish(_pose(0.10, 1.0))
    await bus.publish(_pose(0.01, 1.1))
    await bus.publish(_pose(0.01, 1.2))
    await bus.publish(_pose(0.10, 1.3))

    assert received == ["idle", "press", "idle", "release"]
