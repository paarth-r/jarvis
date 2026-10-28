"""GestureHCIModule + GestureEngine: one gesture at a time.

Regression for the reported bug: making a fist clicked / dragged (the thumb
brushes the index while the hand closes) and sometimes opened Spotlight (the
index reads as pointing for a frame on the way in or out)."""
import numpy as np
import pytest

from jarvis.events import GestureEvent, WorldPoseEvent
from jarvis.modules.gesture_hci import PinchFSM
from jarvis.runtime.event_bus import EventBus
from tests.hand_synth import FIST, OPEN, POINT, hand, pinch
from tests.test_gesture_hci import _BOX_CFG, _make_module, _tp_pose

FPS = 30.0


def _ev(t, shape, x=0.0, pinch_m=0.10):
    e = _tp_pose(x, t, pinch=pinch_m)
    return WorldPoseEvent(landmarks_3d=e.landmarks_3d, timestamp=t, confidence=1.0,
                          hand_side="right", hand_shape=shape)


async def _setup(monkeypatch, **cfg):
    bus = EventBus()
    gestures = []

    async def on_gesture(e: GestureEvent):
        gestures.append((e.name, e.phase))

    bus.subscribe(GestureEvent, on_gesture)
    controller = await _make_module(bus, monkeypatch, **{**_BOX_CFG, **cfg})
    return bus, controller, gestures


def _kinds(controller):
    return [c[0] for c in controller.calls]


@pytest.mark.asyncio
async def test_closing_a_fist_never_clicks_or_drags(monkeypatch):
    bus, controller, gestures = await _setup(monkeypatch)
    t = 0.0
    await bus.publish(_ev(t, hand(OPEN)))
    # Closing: the thumb brushes the index (pinch distance tiny) a frame
    # before the shape reads as a fist; then a 1 s fist with thumb on index.
    t += 1 / FPS
    await bus.publish(_ev(t, pinch(), pinch_m=0.01))
    for _ in range(30):
        t += 1 / FPS
        await bus.publish(_ev(t, hand(FIST), pinch_m=0.01))
    for _ in range(5):                       # past the 0.1 s release grace
        t += 1 / FPS
        await bus.publish(_ev(t, hand(OPEN)))
    kinds = _kinds(controller)
    assert "click" not in kinds and "mouse_down" not in kinds
    assert ("fist", "start") in gestures and ("fist", "end") in gestures


@pytest.mark.asyncio
async def test_index_flicker_on_the_way_into_a_fist_is_not_a_point(monkeypatch):
    bus, controller, gestures = await _setup(monkeypatch)
    t = 0.0
    for shape in [hand(OPEN), hand(POINT), hand(POINT)] + [hand(FIST)] * 10 + [hand(POINT)] * 2 + [hand(OPEN)] * 5:
        await bus.publish(_ev(t, shape))
        t += 1 / FPS
    assert [g for g in gestures if g[0] == "point"] == []


@pytest.mark.asyncio
async def test_held_point_starts_once_and_ends(monkeypatch):
    bus, controller, gestures = await _setup(monkeypatch)
    t = 0.0
    for _ in range(15):                      # 0.5 s of pointing
        await bus.publish(_ev(t, hand(POINT)))
        t += 1 / FPS
    for _ in range(6):
        await bus.publish(_ev(t, hand(OPEN)))
        t += 1 / FPS
    assert gestures == [("point", "start"), ("point", "end")]


@pytest.mark.asyncio
async def test_quick_pinch_still_clicks_immediately(monkeypatch):
    bus, controller, gestures = await _setup(monkeypatch)
    await bus.publish(_ev(0.0, hand(OPEN)))
    await bus.publish(_ev(0.05, pinch(), pinch_m=0.01))
    await bus.publish(_ev(0.10, hand(OPEN)))
    assert _kinds(controller).count("click") == 1
    assert gestures == []


@pytest.mark.asyncio
async def test_fist_holds_the_cursor_and_it_does_not_jump_after(monkeypatch):
    bus, controller, gestures = await _setup(monkeypatch)
    await bus.publish(_ev(0.0, hand(OPEN), x=0.0))
    for i in range(1, 12):                   # fist while the hand drifts 5.5 cm
        await bus.publish(_ev(i / FPS, hand(FIST), x=0.005 * i))
    await bus.publish(_ev(12 / FPS, hand(OPEN), x=0.055))
    await bus.publish(_ev(16 / FPS, hand(OPEN), x=0.055))
    moves = [c for c in controller.calls if c[0] == "move"]
    assert all(m[1:] == pytest.approx((0.5, 0.5), abs=1e-3) for m in moves)


@pytest.mark.asyncio
async def test_opening_a_fist_with_thumb_on_index_does_not_click(monkeypatch):
    bus, controller, gestures = await _setup(monkeypatch)
    t = 0.0
    for _ in range(10):
        await bus.publish(_ev(t, hand(FIST), pinch_m=0.01))
        t += 1 / FPS
    # Hand opens but the thumb is still near the index for a few frames, then
    # separates: that's the fist ending, not a pinch.
    for d in (0.01, 0.01, 0.01, 0.01, 0.05, 0.10):
        await bus.publish(_ev(t, pinch(), pinch_m=d))
        t += 1 / FPS
    assert "click" not in _kinds(controller)
    # A real pinch afterwards works again.
    await bus.publish(_ev(t, pinch(), pinch_m=0.01))
    await bus.publish(_ev(t + 0.05, hand(OPEN)))
    assert _kinds(controller).count("click") == 1


@pytest.mark.asyncio
async def test_no_hand_shape_keeps_legacy_behaviour(monkeypatch):
    # Producers without hand_shape (old replays) see no gesture arbitration.
    bus, controller, gestures = await _setup(monkeypatch)
    await bus.publish(_tp_pose(0.0, 0.0))
    await bus.publish(_tp_pose(0.0, 0.05, pinch=0.01))
    await bus.publish(_tp_pose(0.0, 0.10))
    assert _kinds(controller).count("click") == 1 and gestures == []


# --- fist-tilt scroll ---

from tests.hand_synth import rotation  # noqa: E402

_SCROLL = {"scroll": {"smoothing": 1.0, "curve": 1.0, "deadzone_deg": 8,
                      "full_deg": 35, "max_px_s": 3000}}


def _fist(pitch_deg=0.0):
    return hand(FIST, R=rotation(pitch=np.radians(pitch_deg)))


def _scrolled(controller):
    return sum(c[2] for c in controller.calls if c[0] == "scroll_px")


@pytest.mark.asyncio
async def test_fist_then_tilt_scrolls_and_opening_stops_it(monkeypatch):
    bus, controller, gestures = await _setup(monkeypatch, **_SCROLL)
    t = 0.0
    for _ in range(6):                       # fist, level: activates, no scroll
        await bus.publish(_ev(t, _fist(0)))
        t += 1 / FPS
    assert _scrolled(controller) == 0
    for _ in range(30):                      # tip forward 25 deg for 1 s
        await bus.publish(_ev(t, _fist(25)))
        t += 1 / FPS
    down = _scrolled(controller)
    assert 1500 < down < 3000
    for _ in range(10):                      # open the hand
        await bus.publish(_ev(t, hand(OPEN)))
        t += 1 / FPS
    after = _scrolled(controller)
    assert after - down < 300                # only the release-grace frames
    for _ in range(10):
        await bus.publish(_ev(t, hand(OPEN)))
        t += 1 / FPS
    assert _scrolled(controller) == after


@pytest.mark.asyncio
async def test_scroll_angle_is_measured_from_the_fist_start(monkeypatch):
    bus, controller, gestures = await _setup(monkeypatch, **_SCROLL)
    t = 0.0
    for _ in range(40):                      # fist made already tipped 25 deg
        await bus.publish(_ev(t, _fist(25)))
        t += 1 / FPS
    assert _scrolled(controller) == 0


@pytest.mark.asyncio
async def test_scroll_can_be_disabled(monkeypatch):
    bus, controller, gestures = await _setup(monkeypatch, scroll={"enabled": False})
    t = 0.0
    for p in [0] * 6 + [25] * 30:
        await bus.publish(_ev(t, _fist(p)))
        t += 1 / FPS
    assert _scrolled(controller) == 0
    assert ("fist", "start") in gestures


# --- PinchFSM lock ---

def test_locked_fsm_ignores_pinches_until_the_thumb_opens():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04, hold_delay=0.2)
    assert fsm.lock() == "idle"
    assert fsm.update(0.01, 1.0) == "idle"
    assert fsm.update(0.01, 1.5) == "idle"          # never presses while locked
    assert fsm.update(0.05, 1.6) == "idle"          # thumb opens: unlocked
    assert fsm.state == "open"
    fsm.update(0.01, 1.7)
    assert fsm.update(0.05, 1.8) == "click"


def test_lock_while_dragging_reports_release():
    fsm = PinchFSM(pinch_thresh=0.02, release_thresh=0.04, hold_delay=0.2)
    fsm.update(0.01, 1.0)
    fsm.update(0.01, 1.3)
    assert fsm.state == "dragging"
    assert fsm.lock() == "release"
