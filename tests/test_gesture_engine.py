import numpy as np
import pytest

from jarvis.gestures.builtins import is_fist, is_pointing
from jarvis.gestures.engine import GestureArbiter, GestureEngine
from tests.hand_synth import FIST, OPEN, PEACE, POINT, hand, pinch, rotation


# --- built-in shape rules on synthetic hands ---

@pytest.mark.parametrize("R", [None, rotation(0.5, -0.3, 1.0)])
def test_builtins_on_synthetic_hands(R):
    assert is_fist(hand(FIST, R=R)) and not is_pointing(hand(FIST, R=R))
    assert is_pointing(hand(POINT, R=R)) and not is_fist(hand(POINT, R=R))
    assert not is_fist(hand(OPEN, R=R)) and not is_pointing(hand(OPEN, R=R))
    assert not is_fist(pinch(R=R)) and not is_pointing(pinch(R=R))


# --- arbiter: one active gesture, dwell to start, grace to end ---

def _arb(**kw):
    return GestureArbiter(dwell_s=0.15, release_s=0.1, **kw)


def test_candidate_must_be_held_for_dwell():
    a = _arb()
    assert a.update("peace", 0.0) == (None, None)
    assert a.update("peace", 0.1) == (None, None)
    assert a.update("peace", 0.16) == ("peace", None)
    assert a.active == "peace"


def test_a_brief_flicker_never_activates():
    # The index straightening for two frames on the way into a fist.
    a = _arb()
    a.update("point", 0.0)
    a.update("point", 0.033)
    a.update("fist", 0.066)
    assert a.active is None
    a.update("fist", 0.25)
    assert a.active == "fist"


def test_switching_candidates_restarts_the_dwell():
    a = _arb()
    a.update("peace", 0.0)
    a.update("rock", 0.1)
    assert a.update("rock", 0.2) == (None, None)     # only 0.1 s of rock
    assert a.update("rock", 0.26) == ("rock", None)


def test_active_gesture_survives_short_dropouts_and_blocks_others():
    a = _arb()
    a.update("fist", 0.0)
    a.update("fist", 0.2)
    assert a.update(None, 0.25) == (None, None)       # 50 ms dropout: still fist
    assert a.update("point", 0.28) == (None, None)    # others can't start
    assert a.update("fist", 0.3) == (None, None)
    assert a.active == "fist"


def test_active_gesture_ends_after_release_grace():
    a = _arb()
    a.update("fist", 0.0)
    a.update("fist", 0.2)
    a.update(None, 0.3)
    assert a.update(None, 0.41) == (None, "fist")
    assert a.active is None


def test_next_gesture_needs_its_own_dwell_after_one_ends():
    a = _arb()
    a.update("fist", 0.0)
    a.update("fist", 0.2)
    a.update("point", 0.3)
    assert a.update("point", 0.41) == (None, "fist")
    assert a.update("point", 0.5) == (None, None)
    assert a.update("point", 0.57) == ("point", None)


def test_per_gesture_dwell_overrides():
    a = _arb(dwell={"fist": 0.05, "point": 0.3})
    a.update("fist", 0.0)
    assert a.update("fist", 0.06) == ("fist", None)
    b = _arb(dwell={"fist": 0.05, "point": 0.3})
    b.update("point", 0.0)
    assert b.update("point", 0.2) == (None, None)
    assert b.update("point", 0.31) == ("point", None)


def test_reset_ends_the_active_gesture():
    a = _arb()
    a.update("fist", 0.0)
    a.update("fist", 0.2)
    assert a.reset() == "fist"
    assert a.active is None and a.reset() is None


# --- engine: classification + arbitration ---

def _engine(**kw):
    return GestureEngine({"dwell_s": 0.15, "release_s": 0.1,
                          "dwell": {"fist": 0.1, "point": 0.3}, **kw})


def test_engine_activates_fist_and_reports_transitions():
    e = _engine()
    out = [e.update(hand(FIST), "open", t / 30.0) for t in range(6)]
    assert [f.started for f in out].count("fist") == 1
    assert out[-1].active == "fist"
    assert out[0].candidate == "fist"


def test_engine_point_needs_longer_dwell():
    e = _engine()
    frames = [e.update(hand(POINT), "open", t / 30.0) for t in range(8)]   # 0.23 s
    assert all(f.active is None for f in frames)
    assert e.update(hand(POINT), "open", 0.31).started == "point"


def test_engine_no_shape_means_no_gesture():
    e = _engine()
    assert e.update(None, "open", 0.0).candidate is None


def test_pending_pinch_only_lets_fist_through():
    e = _engine()
    assert e.update(hand(POINT), "pending", 0.0).candidate is None
    assert e.update(hand(FIST), "pending", 0.1).candidate == "fist"


def test_dragging_suppresses_every_gesture():
    e = _engine()
    assert e.update(hand(FIST), "dragging", 0.0).candidate is None


def test_engine_open_hand_and_peace_are_not_builtins():
    e = _engine()
    assert e.update(hand(OPEN), "open", 0.0).candidate is None
    assert e.update(hand(PEACE), "open", 0.1).candidate is None
