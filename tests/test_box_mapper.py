import numpy as np
import pytest

from jarvis.modules.box_mapper import BoxMapper


def _mapper(**kw):
    # filter effectively off so tests see raw geometry
    base = dict(box_width_m=0.15, box_height_m=0.10, min_cutoff=1e6, beta=0.0)
    base.update(kw)
    return BoxMapper(**base)


def test_first_sight_maps_to_screen_centre_wherever_the_hand_is():
    assert _mapper().update(np.array([0.3, -0.2]), 0.0) == (0.5, 0.5)


def test_box_width_of_travel_spans_the_screen_linearly():
    m = _mapper()
    m.update(np.array([0.0, 0.0]), 0.0)
    x_right, _ = m.update(np.array([-0.075, 0.0]), 0.03)   # hand moves image-left
    assert x_right == pytest.approx(1.0, abs=1e-4)
    x_mid, _ = m.update(np.array([0.0, 0.0]), 0.06)
    assert x_mid == pytest.approx(0.5, abs=1e-4)
    x_q, _ = m.update(np.array([0.0375, 0.0]), 0.09)
    assert x_q == pytest.approx(0.25, abs=1e-4)                        # linear, mirrored


def test_vertical_uses_box_height_and_is_not_mirrored():
    m = _mapper()
    m.update(np.array([0.0, 0.0]), 0.0)
    _, y = m.update(np.array([0.0, 0.025]), 0.03)
    assert y == pytest.approx(0.75, abs=1e-4)


def test_edge_push_slides_box_so_reversal_comes_straight_back():
    m = _mapper()
    m.update(np.array([0.0, 0.0]), 0.0)
    m.update(np.array([-0.075, 0.0]), 0.03)                 # at right edge
    x, _ = m.update(np.array([-0.175, 0.0]), 0.06)          # 10 cm past it
    assert x == pytest.approx(1.0, abs=1e-4)
    x, _ = m.update(np.array([-0.175 + 0.0375, 0.0]), 0.09)  # back 3.75 cm
    assert x == pytest.approx(0.75, abs=1e-4)                          # not still pinned


def test_reacquire_keeps_cursor_where_it_was():
    m = _mapper()
    m.update(np.array([0.0, 0.0]), 0.0)
    before = m.update(np.array([-0.03, 0.01]), 0.03)
    after = m.update(np.array([0.4, 0.4]), 1.0)              # hand lost, returns elsewhere
    assert after == pytest.approx(before, abs=1e-4)


def test_filter_holds_still_hand_steady():
    rng = np.random.default_rng(0)
    m = BoxMapper(min_cutoff=0.8, beta=10.0)
    xs = [m.update(rng.normal(0.0, 0.001, 2), i / 30.0)[0] for i in range(120)]
    raw_px = 0.001 / 0.15 * 2560          # 1 mm jitter ~ 17 px on a 2560 screen
    assert np.std(xs[30:]) * 2560 < 0.5 * raw_px
