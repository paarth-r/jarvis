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

    def move_by(self, dx, dy):
        self.calls.append(("move_by", dx, dy))

    def click_here(self):
        self.calls.append(("click_here",))

    def mouse_down_here(self):
        self.calls.append(("mouse_down_here",))

    def mouse_up_here(self):
        self.calls.append(("mouse_up_here",))

    def scroll_px(self, dx, dy):
        self.calls.append(("scroll_px", dx, dy))


def _pose(pinch_dist: float, ts: float) -> WorldPoseEvent:
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 2] = 0.5
    lm[INDEX_TIP, 0] = pinch_dist  # thumb tip stays at x=0
    return WorldPoseEvent(landmarks_3d=lm, timestamp=ts, confidence=1.0, hand_side="right")


async def _make_module(bus, monkeypatch, **extra):
    controller = FakeController()
    import mouse_control.control
    monkeypatch.setattr(mouse_control.control, "MouseController", lambda **kw: controller)
    mod = GestureHCIModule("gesture_hci", {
        "pinch_thresh_m": 0.02,
        "release_thresh_m": 0.04,
        "drag_hold_s": 0.2,
        **extra,
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


# --- trackpad mode: relative motion, fist clutch, pinch freeze ---

from jarvis.modules.pointing import (
    INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP, MIDDLE_TIP, RING_TIP, PINKY_TIP,
)
from jarvis.modules.gesture_hci import THUMB_MCP

_TIPS_MCPS = [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
              (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)]


def _tp_pose(x: float, ts: float, pinch: float = 0.10, fist: bool = False) -> WorldPoseEvent:
    """Wrist at (x, 0); knuckles 8 cm up; tips 16 cm up (open) or 4 cm (fist).
    Thumb tip sits `pinch` metres from the index tip."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 2] = 0.5
    lm[:, 0] = x
    tip_y = -0.04 if fist else -0.16
    for tip, mcp in _TIPS_MCPS:
        lm[mcp, 1] = -0.08
        lm[tip, 1] = tip_y
    lm[THUMB_MCP, 1] = -0.08
    lm[THUMB_TIP] = lm[INDEX_TIP] + np.array([pinch, 0, 0], dtype=np.float32)
    return WorldPoseEvent(landmarks_3d=lm, timestamp=ts, confidence=1.0, hand_side="right")


_TP_CFG = {"mode": "trackpad",
           "trackpad": {"min_cutoff": 1e6, "beta": 0.0}}   # filter off for geometry


def _moved(controller):
    return sum(c[1] for c in controller.calls if c[0] == "move_by")


@pytest.mark.asyncio
async def test_trackpad_moves_relatively(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch, **_TP_CFG)
    for i in range(10):
        await bus.publish(_tp_pose(0.3 + 0.002 * i, i / 30.0))   # hand far right
    assert _moved(controller) < 0                                # mirrored X
    assert "move" not in [c[0] for c in controller.calls]       # no absolute moves


@pytest.mark.asyncio
async def test_fist_clutch_repositions_without_moving_cursor(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch, **_TP_CFG)
    await bus.publish(_tp_pose(0.0, 0.0))
    for i in range(1, 10):                                       # fist, slide 9 cm
        await bus.publish(_tp_pose(0.01 * i, i / 30.0, fist=True))
    await bus.publish(_tp_pose(0.09, 10 / 30.0))                 # open again
    await bus.publish(_tp_pose(0.09, 11 / 30.0))
    assert _moved(controller) == 0
    kinds = [c[0] for c in controller.calls]
    assert "click_here" not in kinds and "mouse_down_here" not in kinds


@pytest.mark.asyncio
async def test_cursor_frozen_while_pinch_pending_then_clicks_here(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch, **_TP_CFG)
    await bus.publish(_tp_pose(0.0, 0.0))
    await bus.publish(_tp_pose(0.005, 0.05, pinch=0.01))   # closes while knuckles shift
    await bus.publish(_tp_pose(0.010, 0.10, pinch=0.01))
    await bus.publish(_tp_pose(0.010, 0.15))               # opens < 0.2s — click
    kinds = [c[0] for c in controller.calls]
    assert kinds == ["click_here"]                          # no motion before it


@pytest.mark.asyncio
async def test_trackpad_drag_moves_with_button_down(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch, **_TP_CFG)
    await bus.publish(_tp_pose(0.0, 0.0))
    await bus.publish(_tp_pose(0.0, 0.05, pinch=0.01))     # pending
    await bus.publish(_tp_pose(0.0, 0.30, pinch=0.01))     # press
    for i in range(1, 6):
        await bus.publish(_tp_pose(0.004 * i, 0.30 + i / 30.0, pinch=0.01))
    await bus.publish(_tp_pose(0.02, 0.6))                 # open — release
    kinds = [c[0] for c in controller.calls]
    down, up = kinds.index("mouse_down_here"), kinds.index("mouse_up_here")
    assert "move_by" in kinds[down:up]


@pytest.mark.asyncio
async def test_fist_during_drag_releases_button(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch, **_TP_CFG)
    await bus.publish(_tp_pose(0.0, 0.0, pinch=0.01))
    await bus.publish(_tp_pose(0.0, 0.25, pinch=0.01))     # press
    await bus.publish(_tp_pose(0.0, 0.30, fist=True))
    kinds = [c[0] for c in controller.calls]
    assert kinds == ["mouse_down_here", "mouse_up_here"]


# --- box mode: absolute from a small hand box, pinch freeze ---

_BOX_CFG = {"mode": "box", "box": {"min_cutoff": 1e6, "beta": 0.0}}


def _last_move(controller):
    return [c for c in controller.calls if c[0] == "move"][-1]


@pytest.mark.asyncio
async def test_box_small_travel_spans_screen(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch, **_BOX_CFG)
    await bus.publish(_tp_pose(0.30, 0.0))                  # first sight -> centre
    assert _last_move(controller)[1:] == pytest.approx((0.5, 0.5))
    await bus.publish(_tp_pose(0.30 - 0.075, 0.03))         # 7.5 cm -> right edge
    assert _last_move(controller)[1] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.asyncio
async def test_box_click_lands_where_pinch_closed(monkeypatch):
    bus = EventBus()
    controller = await _make_module(bus, monkeypatch, **_BOX_CFG)
    await bus.publish(_tp_pose(0.0, 0.0))
    await bus.publish(_tp_pose(0.006, 0.05, pinch=0.01))    # closes, knuckles shift
    await bus.publish(_tp_pose(0.006, 0.10))                # opens — click
    click = [c for c in controller.calls if c[0] == "click"]
    assert len(click) == 1
    assert click[0][1:] == pytest.approx((0.5, 0.5), abs=1e-3)
    await bus.publish(_tp_pose(0.006, 0.13))                # no jump afterwards
    assert _last_move(controller)[1:] == pytest.approx((0.5, 0.5), abs=1e-3)
