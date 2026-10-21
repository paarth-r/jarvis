import numpy as np

from jarvis.modules.gaze_model import (
    EYE_A_INNER, EYE_A_IRIS, EYE_A_OUTER, EYE_B_INNER, EYE_B_IRIS, EYE_B_OUTER,
    GazeMapper, gaze_features,
)


def _face(iris_frac_x: float, iris_frac_y: float = 0.0) -> np.ndarray:
    """Two horizontal eyes 0.1 wide; irises placed at the given fraction."""
    lm = np.zeros((478, 3))
    lm[EYE_A_OUTER, :2] = (0.30, 0.40)
    lm[EYE_A_INNER, :2] = (0.40, 0.40)
    lm[EYE_B_INNER, :2] = (0.60, 0.40)
    lm[EYE_B_OUTER, :2] = (0.70, 0.40)
    lm[EYE_A_IRIS, :2] = (0.30 + 0.1 * iris_frac_x, 0.40 + 0.1 * iris_frac_y)
    lm[EYE_B_IRIS, :2] = (0.60 + 0.1 * iris_frac_x, 0.40 + 0.1 * iris_frac_y)
    return lm


def test_centered_iris_gives_half():
    f = gaze_features(_face(0.5))
    assert np.isclose(f[0], 0.5)
    assert np.isclose(f[1], 0.0)


def test_iris_offsets_track_both_axes():
    assert gaze_features(_face(0.7))[0] > gaze_features(_face(0.3))[0]
    # perpendicular axis: image y down -> positive offset
    assert gaze_features(_face(0.5, 0.2))[1] > gaze_features(_face(0.5, -0.2))[1]


def test_features_invariant_to_face_scale():
    small = _face(0.6, 0.1)
    big = small.copy()
    big[:, :2] = 0.5 + (small[:, :2] - 0.5) * 2.0
    assert np.allclose(gaze_features(small)[:2], gaze_features(big)[:2])


def test_head_matrix_identity_is_zero_pose():
    f = gaze_features(_face(0.5), np.eye(4))
    assert np.isclose(f[2], 0.0) and np.isclose(f[3], 0.0)


def test_mapper_recovers_a_known_mapping():
    rng = np.random.default_rng(0)
    feats = rng.uniform(0, 1, size=(200, 6))
    targets = np.column_stack([
        0.2 + 0.6 * feats[:, 0] + 0.1 * feats[:, 2],
        0.5 * feats[:, 1] + 0.2 * feats[:, 0] ** 2,
    ])
    m = GazeMapper.fit(feats, targets, ridge=1e-8)
    assert np.allclose(m.predict(feats), targets, atol=1e-6)
    assert m.predict(feats[0]).shape == (2,)


def test_mapper_round_trips_through_json(tmp_path):
    rng = np.random.default_rng(1)
    feats = rng.uniform(0, 1, size=(50, 6))
    targets = rng.uniform(0, 1, size=(50, 2))
    m = GazeMapper.fit(feats, targets)
    p = tmp_path / "calib.json"
    m.save(p, extra={"screen": [1512, 982]})
    m2 = GazeMapper.load(p)
    assert np.allclose(m.predict(feats), m2.predict(feats))
