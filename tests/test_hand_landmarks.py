import numpy as np
import pytest
from jarvis.modules.hand_landmarks import (
    HAND_CONNECTIONS, hand_points, draw_landmarks,
)


def test_hand_connections_are_valid_indices():
    # 21 landmarks → all endpoints in range, thumb+4 fingers+palm present
    assert len(HAND_CONNECTIONS) >= 20
    for a, b in HAND_CONNECTIONS:
        assert 0 <= a < 21 and 0 <= b < 21


def test_hand_points_maps_axes_x_z_negy():
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.1, 0.2, 0.5]   # world: x right, y down, z forward
    pts = hand_points(lm)
    assert len(pts) == 21
    # display = (x, z, -y)
    assert pts[0] == pytest.approx((0.1, 0.5, -0.2), abs=1e-6)


def test_hand_points_none_returns_21_nones():
    pts = hand_points(None)
    assert pts == [None] * 21


def test_draw_landmarks_mono_frame_returns_bgr():
    frame = np.zeros((800, 1280), dtype=np.uint8)  # mono
    lm2d = np.zeros((21, 3), dtype=np.float32)
    lm2d[:, 0] = 0.5
    lm2d[:, 1] = 0.5
    out = draw_landmarks(frame, lm2d, HAND_CONNECTIONS)
    assert out.shape == (800, 1280, 3)
    assert out.dtype == np.uint8


def test_draw_landmarks_does_not_mutate_input_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    before = frame.copy()
    lm2d = np.full((21, 3), 0.5, dtype=np.float32)
    draw_landmarks(frame, lm2d, HAND_CONNECTIONS)
    assert np.array_equal(frame, before)
