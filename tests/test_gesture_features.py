import numpy as np
import pytest

from jarvis.gestures.features import FEATURE_DIM, FEATURE_NAMES, features, palm_frame, tilt
from tests.hand_synth import FIST, OPEN, PEACE, POINT, ROCK, hand, rotation


def test_feature_vector_shape_and_names():
    f = features(hand(OPEN))
    assert f.shape == (FEATURE_DIM,)
    assert len(FEATURE_NAMES) == FEATURE_DIM


@pytest.mark.parametrize("pose", [OPEN, FIST, POINT, PEACE])
def test_invariant_to_rotation_scale_translation(pose):
    base = features(hand(pose))
    moved = features(hand(pose, R=rotation(0.7, -0.4, 1.2), scale=1.3, offset=(0.2, -0.1, 0.5)))
    np.testing.assert_allclose(moved, base, atol=1e-4)


def test_left_and_right_hands_match():
    np.testing.assert_allclose(features(hand(PEACE, mirror=True)), features(hand(PEACE)), atol=1e-4)


def test_open_hand_has_straight_fingers_fist_does_not():
    names = FEATURE_NAMES
    pip = [names.index(f"{f}_pip") for f in ("index", "middle", "ring", "pinky")]
    assert np.all(features(hand(OPEN))[pip] < 0.05)
    assert np.all(features(hand(FIST))[pip] > 1.2)


def test_noise_moves_features_less_than_a_different_pose():
    rng = np.random.default_rng(1)
    base = features(hand(PEACE))
    noisy = [np.linalg.norm(features(hand(PEACE, noise=0.002, rng=rng)) - base) for _ in range(20)]
    for other in (ROCK, POINT, FIST, OPEN):
        assert max(noisy) < np.linalg.norm(features(hand(other)) - base)


def test_degenerate_hand_does_not_crash():
    f = features(np.zeros((21, 3), dtype=np.float32))
    assert np.all(np.isfinite(f))


def test_palm_frame_is_orthonormal_and_follows_rotation():
    R = rotation(0.3, 0.2, -0.5)
    F0 = palm_frame(hand(FIST))
    F1 = palm_frame(hand(FIST, R=R))
    np.testing.assert_allclose(F1.T @ F1, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(F1, R @ F0, atol=1e-5)


def test_tilt_recovers_pitch_and_roll_relative_to_reference():
    ref = palm_frame(hand(FIST))
    pitch, roll = tilt(ref, palm_frame(hand(FIST, R=rotation(pitch=np.radians(20)))))
    assert pitch == pytest.approx(20, abs=0.5)
    assert roll == pytest.approx(0, abs=0.5)
    pitch, roll = tilt(ref, palm_frame(hand(FIST, R=rotation(roll=np.radians(-15)))))
    assert pitch == pytest.approx(0, abs=0.5)
    assert roll == pytest.approx(-15, abs=0.5)


def test_tilt_is_relative_to_a_rotated_start():
    # Started with the fist already turned; tilting from there reads as the delta.
    R0 = rotation(0.4, 0.3, 0.2)
    ref = palm_frame(hand(FIST, R=R0))
    pitch, _ = tilt(ref, palm_frame(hand(FIST, R=R0 @ rotation(pitch=np.radians(-12)))))
    assert pitch == pytest.approx(-12, abs=0.5)
