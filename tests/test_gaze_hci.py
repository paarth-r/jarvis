import numpy as np
import pytest

from jarvis.events import GazeEvent, WorldPoseEvent
from jarvis.modules.gaze_hci import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, GazeHCIModule, PinchPair
from jarvis.runtime.event_bus import EventBus

OPEN, PINCH = 0.08, 0.01


class FakePointer:
    size = (1000.0, 1000.0)

    def __init__(self):
        self.events = []

    def post(self, kind, nx, ny):
        self.events.append((kind, round(nx, 3), round(ny, 3)))


def _hand(d_index, d_middle):
    lm = np.zeros((21, 3))
    lm[INDEX_TIP] = (d_index, 0, 0)
    lm[MIDDLE_TIP] = (0, d_middle, 0)
    return WorldPoseEvent(landmarks_3d=lm, timestamp=0.0, confidence=1.0, hand_side="right")


async def _module(**cfg):
    ptr = FakePointer()
    m = GazeHCIModule("gaze_hci", {"pinch_thresh_m": 0.025, "release_thresh_m": 0.045,
                                   "drag_hold_s": 0.3, **cfg}, pointer=ptr)
    await m.setup(EventBus())
    return m, ptr


async def _pose(m, d_index, d_middle, t):
    e = _hand(d_index, d_middle)
    e.timestamp = t
    await m._on_pose(e)


def test_pair_index_quick_pinch_is_left_click():
    p = PinchPair(0.025, 0.045, 0.3)
    assert p.update(PINCH, OPEN, 0.0) == ("left", "idle")
    assert p.update(OPEN, OPEN, 0.1) == ("left", "click")
    assert not p.active


def test_pair_middle_pinch_is_right_click_even_when_held():
    p = PinchPair(0.025, 0.045, 0.3)
    p.update(OPEN, PINCH, 0.0)
    assert p.update(OPEN, PINCH, 5.0) == ("right", "idle")   # never becomes a drag
    assert p.update(OPEN, OPEN, 5.1) == ("right", "click")


def test_pair_rolling_fingers_keeps_the_first_owner():
    p = PinchPair(0.025, 0.045, 0.3)
    p.update(PINCH, OPEN, 0.0)                     # index claims it
    assert p.update(0.03, PINCH, 0.1) == ("left", "idle")  # middle now closer: ignored
    assert p.update(OPEN, OPEN, 0.2) == ("left", "click")


async def test_follow_mode_moves_cursor_to_gaze():
    m, ptr = await _module()
    await m._on_gaze(GazeEvent(0.2, 0.3, 0.0, 1.0))
    assert ptr.events == [("move", 0.2, 0.3)]


async def test_follow_mode_ignores_sub_threshold_jitter():
    m, ptr = await _module(move_threshold_px=5.0)
    await m._on_gaze(GazeEvent(0.2, 0.3, 0.0, 1.0))
    await m._on_gaze(GazeEvent(0.202, 0.3, 0.1, 1.0))   # 2px on a 1000px screen
    assert len(ptr.events) == 1


async def test_cursor_freezes_while_a_click_forms():
    m, ptr = await _module()
    await m._on_gaze(GazeEvent(0.2, 0.3, 0.0, 1.0))
    await _pose(m, PINCH, OPEN, 0.0)
    await m._on_gaze(GazeEvent(0.6, 0.6, 0.05, 1.0))    # eyes flick away mid-pinch
    await _pose(m, OPEN, OPEN, 0.1)
    assert ptr.events == [("move", 0.2, 0.3), ("left_down", 0.2, 0.3), ("left_up", 0.2, 0.3)]


async def test_right_click_from_middle_pinch():
    m, ptr = await _module()
    await m._on_gaze(GazeEvent(0.5, 0.5, 0.0, 1.0))
    await _pose(m, OPEN, PINCH, 0.0)
    await _pose(m, OPEN, OPEN, 0.1)
    assert ptr.events[-2:] == [("right_down", 0.5, 0.5), ("right_up", 0.5, 0.5)]


async def test_held_pinch_drags_along_gaze():
    m, ptr = await _module()
    await m._on_gaze(GazeEvent(0.1, 0.1, 0.0, 1.0))
    await _pose(m, PINCH, OPEN, 0.0)
    await _pose(m, PINCH, OPEN, 0.4)                     # past drag_hold_s
    await m._on_gaze(GazeEvent(0.5, 0.5, 0.5, 1.0))
    await _pose(m, OPEN, OPEN, 0.6)
    assert ptr.events == [("move", 0.1, 0.1), ("left_down", 0.1, 0.1),
                          ("drag", 0.5, 0.5), ("left_up", 0.5, 0.5)]


async def test_jump_on_pinch_leaves_cursor_until_pinch():
    m, ptr = await _module(jump_on_pinch=True)
    await m._on_gaze(GazeEvent(0.2, 0.3, 0.0, 1.0))
    await m._on_gaze(GazeEvent(0.7, 0.8, 0.1, 1.0))
    assert ptr.events == []
    await _pose(m, PINCH, OPEN, 0.2)
    await _pose(m, OPEN, OPEN, 0.3)
    assert ptr.events == [("move", 0.7, 0.8), ("left_down", 0.7, 0.8), ("left_up", 0.7, 0.8)]


async def test_no_gaze_yet_means_no_click():
    m, ptr = await _module()
    await _pose(m, PINCH, OPEN, 0.0)
    await _pose(m, OPEN, OPEN, 0.1)
    assert ptr.events == []
