import numpy as np
import pytest

from jarvis.gestures.features import palm_frame
from jarvis.gestures.tilt_scroll import TiltScroller
from tests.hand_synth import FIST, hand, rotation


def _frame(pitch_deg=0.0, roll_deg=0.0):
    return palm_frame(hand(FIST, R=rotation(pitch=np.radians(pitch_deg), roll=np.radians(roll_deg))))


def _run(scroller, pitch, roll=0.0, seconds=1.0, fps=30):
    scroller.start(_frame(), 0.0)
    dx = dy = 0
    for i in range(1, int(seconds * fps) + 1):
        x, y = scroller.update(_frame(pitch, roll), i / fps)
        dx, dy = dx + x, dy + y
    return dx, dy


def _scroller(**kw):
    return TiltScroller(**{"deadzone_deg": 8, "full_deg": 35, "max_px_s": 3000,
                           "curve": 1.0, "smoothing": 1.0, **kw})


def test_holding_still_inside_the_dead_zone_does_not_scroll():
    assert _run(_scroller(), pitch=6) == (0, 0)


def test_tipping_forward_scrolls_down_and_back_scrolls_up():
    _, down = _run(_scroller(), pitch=20)
    _, up = _run(_scroller(), pitch=-20)
    assert down > 0 and up < 0
    assert down == pytest.approx(-up, rel=0.05)


def test_further_tilt_scrolls_faster_up_to_the_cap():
    s = [_run(_scroller(), pitch=p)[1] for p in (15, 25, 35, 60)]
    assert s[0] < s[1] < s[2]
    assert s[3] == pytest.approx(s[2], rel=0.05)          # capped
    assert s[2] == pytest.approx(3000, rel=0.05)          # max_px_s for 1 s


def test_rate_is_linear_past_the_dead_zone_with_curve_1():
    # halfway between dead zone and full tilt -> half speed
    _, dy = _run(_scroller(), pitch=21.5)
    assert dy == pytest.approx(1500, rel=0.05)


def test_horizontal_is_off_by_default():
    dx, _ = _run(_scroller(), pitch=0, roll=25)
    assert dx == 0


def test_horizontal_from_roll_when_enabled():
    dx_r, _ = _run(_scroller(horizontal=True), pitch=0, roll=25)
    dx_l, _ = _run(_scroller(horizontal=True), pitch=0, roll=-25)
    assert dx_r > 0 and dx_l < 0


def test_invert_flips_direction():
    _, dy = _run(_scroller(invert_y=True), pitch=20)
    assert dy < 0


def test_tilt_is_relative_to_the_starting_angle():
    s = _scroller()
    start = palm_frame(hand(FIST, R=rotation(pitch=np.radians(30))))
    s.start(start, 0.0)
    total = sum(s.update(start, i / 30)[1] for i in range(1, 31))
    assert total == 0                                     # started tilted: no scroll


def test_fractional_pixels_accumulate_instead_of_rounding_away():
    # Barely past the dead zone: well under 1 px per frame, but it adds up.
    _, dy = _run(_scroller(), pitch=9, seconds=2.0)
    assert dy > 0


def test_long_frame_gap_does_not_jump():
    s = _scroller()
    s.start(_frame(), 0.0)
    s.update(_frame(30), 1 / 30)
    _, dy = s.update(_frame(30), 5.0)                     # tracking stalled 5 s
    assert abs(dy) <= 3000 * 0.1 + 1


def test_smoothing_eases_in():
    rough = _scroller(smoothing=1.0)
    smooth = _scroller(smoothing=0.3)
    for s in (rough, smooth):
        s.start(_frame(), 0.0)
    assert smooth.update(_frame(30), 1 / 30)[1] < rough.update(_frame(30), 1 / 30)[1]


def test_update_before_start_is_a_no_op():
    assert _scroller().update(_frame(30), 0.1) == (0, 0)
