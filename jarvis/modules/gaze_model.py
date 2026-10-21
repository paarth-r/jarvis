from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np

# MediaPipe Face Landmarker indices (478-point mesh with iris refinement).
# Eye A = the eye on the image's left for an unmirrored webcam frame.
EYE_A_OUTER, EYE_A_INNER, EYE_A_IRIS = 33, 133, 468
EYE_B_INNER, EYE_B_OUTER, EYE_B_IRIS = 362, 263, 473
NOSE_TIP = 1

DEFAULT_CALIBRATION_PATH = Path.home() / ".jarvis" / "gaze_calibration.json"


def _eye_offset(lm: np.ndarray, corner_0: int, corner_1: int, iris: int) -> np.ndarray:
    """Iris position inside one eye, in the eye's own frame: x along the
    corner-to-corner axis (0 = corner_0, 1 = corner_1), y perpendicular to it.
    Both scaled by eye width, so head distance and in-plane roll drop out."""
    c0, c1, p = lm[corner_0, :2], lm[corner_1, :2], lm[iris, :2]
    axis = c1 - c0
    width_sq = float(axis @ axis)
    if width_sq < 1e-12:
        return np.zeros(2)
    perp = np.array([-axis[1], axis[0]])
    rel = p - c0
    return np.array([rel @ axis / width_sq, rel @ perp / width_sq])


def gaze_features(landmarks: np.ndarray, head_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """Feature vector for one face: [iris_x, iris_y, yaw, pitch, nose_x, nose_y].

    iris_* is the mean in-eye iris offset over both eyes (the eye-in-head
    signal); yaw/pitch come from the face's 4x4 transform when available and
    nose_* is the face's position in the image (the head-in-camera signal).
    A webcam can't separate those cleanly, so the calibration fit uses both.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    a = _eye_offset(lm, EYE_A_OUTER, EYE_A_INNER, EYE_A_IRIS)
    b = _eye_offset(lm, EYE_B_INNER, EYE_B_OUTER, EYE_B_IRIS)
    iris = (a + b) / 2.0

    yaw = pitch = 0.0
    if head_matrix is not None:
        r = np.asarray(head_matrix, dtype=np.float64)[:3, :3]
        yaw = float(np.arctan2(-r[2, 0], np.hypot(r[2, 1], r[2, 2])))
        pitch = float(np.arctan2(r[2, 1], r[2, 2]))

    return np.array([iris[0], iris[1], yaw, pitch, lm[NOSE_TIP, 0], lm[NOSE_TIP, 1]])


def _design(features: np.ndarray) -> np.ndarray:
    """Quadratic in the iris terms (the eyeball is a sphere, not a plane),
    linear in head pose/position, plus a bias column."""
    f = np.atleast_2d(features)
    ix, iy = f[:, 0], f[:, 1]
    return np.column_stack([
        np.ones(len(f)), ix, iy, ix * iy, ix * ix, iy * iy, f[:, 2], f[:, 3], f[:, 4], f[:, 5],
    ])


class GazeMapper:
    """Feature vector -> normalized screen point (0-1, 0-1), fitted from a
    calibration session with ridge-regularized least squares on standardized
    columns (head terms are tiny next to the bias; without scaling the
    ridge would crush them)."""

    def __init__(self, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)   # (n_terms, 2)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, ridge: float = 1e-3) -> "GazeMapper":
        x = _design(np.asarray(features, dtype=np.float64))
        y = np.asarray(targets, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        mean[0], scale[0] = 0.0, 1.0           # keep the bias column as-is
        scale[scale < 1e-9] = 1.0
        xs = (x - mean) / scale
        reg = ridge * len(xs) * np.eye(xs.shape[1])
        reg[0, 0] = 0.0                         # don't shrink the bias
        weights = np.linalg.solve(xs.T @ xs + reg, xs.T @ y)
        return cls(weights, mean, scale)

    def predict(self, features: np.ndarray) -> np.ndarray:
        xs = (_design(features) - self.mean) / self.scale
        out = xs @ self.weights
        return out[0] if np.ndim(features) == 1 else out

    def save(self, path: Path, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            **(extra or {}),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GazeMapper":
        data = json.loads(Path(path).read_text())
        return cls(data["weights"], data["mean"], data["scale"])
