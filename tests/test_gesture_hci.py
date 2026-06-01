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
