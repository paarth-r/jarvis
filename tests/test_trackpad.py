import numpy as np
import pytest

from jarvis.modules.trackpad import TrackpadMapper, accel_gain

FPS = 30.0


def _mapper(**kw):
    # filter effectively off so tests see raw geometry
    base = dict(gain_min=1000.0, gain_max=20000.0, speed_lo=0.05, speed_hi=0.5,
                deadzone_mps=0.005, min_cutoff=1e6, beta=0.0)
    base.update(kw)
    return TrackpadMapper(**base)


def _sweep(m, start, velocity, frames, t0=0.0):
    """Feed a constant-velocity sweep; return summed (dx, dy)."""
    total = np.zeros(2)
    for i in range(frames):
        t = t0 + i / FPS
        total += m.update(np.asarray(start) + np.asarray(velocity) * (t - t0), t)
    return total


def test_gain_is_flat_outside_the_ramp_and_monotonic_inside():
    g = lambda v: accel_gain(v, 1000.0, 20000.0, 0.05, 0.5)
    assert g(0.0) == g(0.05) == 1000.0
    assert g(0.5) == g(2.0) == 20000.0
    samples = [g(v) for v in np.linspace(0.05, 0.5, 20)]
    assert samples == sorted(samples)


def test_first_frame_moves_nothing():
    assert _mapper().update(np.array([0.1, 0.1]), 0.0) == (0, 0)


def test_absolute_hand_position_does_not_matter():
    a = _sweep(_mapper(), [0.0, 0.0], [0.02, 0.0], 10)
    b = _sweep(_mapper(), [0.2, -0.1], [0.02, 0.0], 10)
    assert np.allclose(a, b)


def test_x_is_mirrored_and_y_is_not():
    dx, dy = _sweep(_mapper(), [0, 0], [0.02, 0.02], 10)
    assert dx < 0 and dy > 0


def test_slow_motion_is_precise():
    # 1 cm at 2 cm/s -> gain_min -> ~10 px
    dx, _ = _sweep(_mapper(), [0, 0], [0.02, 0.0], int(0.5 * FPS) + 1)
    assert abs(dx) == pytest.approx(10, abs=1)


def test_fast_motion_reaches_further_per_metre():
    slow = abs(_sweep(_mapper(), [0, 0], [0.02, 0], 16)[0]) / 0.01
    fast = abs(_sweep(_mapper(), [0, 0], [0.6, 0], 3)[0]) / (0.6 * 2 / FPS)
    assert fast > 10 * slow


def test_subpixel_motion_accumulates():
    # 0.2 mm/frame at gain 1000 = 0.2 px/frame: each step alone truncates
    # to 0, but 20 steps carry over to ~4 px
    dx, _ = _sweep(_mapper(), [0, 0], [0.006, 0], 21)
    assert abs(dx) == pytest.approx(4, abs=1)


def test_deadzone_swallows_creep():
    assert tuple(_sweep(_mapper(), [0, 0], [0.002, 0], 30)) == (0, 0)


def test_reset_prevents_jump_on_reacquire():
    m = _mapper()
    m.update(np.array([0.0, 0.0]), 0.0)
    m.reset()
    assert m.update(np.array([0.3, 0.3]), 0.1) == (0, 0)


def test_tracking_gap_does_not_jump():
    m = _mapper()
    m.update(np.array([0.0, 0.0]), 0.0)
    assert m.update(np.array([0.3, 0.3]), 1.0) == (0, 0)   # hand was lost
