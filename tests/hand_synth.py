"""Synthetic MediaPipe-layout hands from joint angles, for gesture tests.

Hand-local frame: fingers point along -y (image up), the palm faces -z (the
camera), x runs pinky -> index for a right hand. Flexion curls a finger toward
+z (into the palm, away from the camera); spread fans it in the palm plane.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Base joint (thumb CMC / finger MCP) positions, metres from the wrist.
_BASE = {
    "thumb": (0.025, -0.025, 0.0),
    "index": (0.030, -0.090, 0.0),
    "middle": (0.010, -0.095, 0.0),
    "ring": (-0.010, -0.090, 0.0),
    "pinky": (-0.028, -0.080, 0.0),
}
_BONES = {
    "thumb": (0.035, 0.030, 0.025),
    "index": (0.040, 0.025, 0.020),
    "middle": (0.045, 0.028, 0.022),
    "ring": (0.042, 0.026, 0.020),
    "pinky": (0.032, 0.020, 0.018),
}
# Resting fan angle (radians, about +z) of each finger's first bone.
_FAN = {"thumb": -0.8, "index": -0.12, "middle": 0.0, "ring": 0.12, "pinky": 0.25}
_START = {"thumb": 1, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

OPEN = {f: (0.0, 0.0, 0.0) for f in FINGERS}
FIST = {"thumb": (0.5, 0.6, 0.5), **{f: (1.5, 1.6, 0.9) for f in FINGERS[1:]}}
POINT = {**FIST, "index": (0.0, 0.0, 0.0)}
PEACE = {**FIST, "index": (0.0, 0.0, 0.0), "middle": (0.0, 0.0, 0.0)}
ROCK = {**FIST, "index": (0.0, 0.0, 0.0), "pinky": (0.0, 0.0, 0.0)}
THREE = {**FIST, "index": (0.0, 0.0, 0.0), "middle": (0.0, 0.0, 0.0), "ring": (0.0, 0.0, 0.0)}


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rotation(pitch: float = 0.0, roll: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """Rotation about x (pitch), then y (roll), then z (yaw), radians."""
    return (_rot(np.array([0, 0, 1.0]), yaw) @ _rot(np.array([0, 1.0, 0]), roll)
            @ _rot(np.array([1.0, 0, 0]), pitch))


def hand(flex: Dict[str, Sequence[float]] = OPEN,
         spread: Dict[str, float] | None = None,
         R: np.ndarray | None = None,
         scale: float = 1.0,
         offset: Sequence[float] = (0.0, 0.0, 0.0),
         mirror: bool = False,
         noise: float = 0.0,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """(21, 3) landmarks for the given per-finger flexion (3 joint angles each)."""
    spread = spread or {}
    pts = np.zeros((21, 3))
    for f in FINGERS:
        p = np.array(_BASE[f], dtype=float)
        pts[_START[f]] = p
        d = _rot(np.array([0, 0, 1.0]), _FAN[f] + spread.get(f, 0.0)) @ np.array([0, -1.0, 0])
        # Flexion axis: in-plane, perpendicular to the finger. Curls toward +z.
        axis = np.cross(d, [0, 0, 1.0])
        if f == "thumb":
            # The thumb curls across the palm rather than straight into it.
            axis = np.cross(d, np.array([0.6, 0.0, 0.8]))
        for k, (length, angle) in enumerate(zip(_BONES[f], flex[f])):
            d = _rot(axis, -angle) @ d
            p = p + length * d
            pts[_START[f] + k + 1] = p
    if mirror:
        pts[:, 0] *= -1
    if R is not None:
        pts = pts @ R.T
    pts = pts * scale + np.asarray(offset, dtype=float)
    if noise:
        rng = rng or np.random.default_rng(0)
        pts = pts + rng.normal(scale=noise, size=pts.shape)
    return pts.astype(np.float32)


def pinch(R: np.ndarray | None = None) -> np.ndarray:
    """Relaxed pinch: index half-curled, thumb tip touching the index tip,
    other fingers loosely open."""
    flex = {**OPEN, "index": (0.6, 0.6, 0.3), "middle": (0.3, 0.3, 0.1),
            "ring": (0.3, 0.3, 0.1), "pinky": (0.3, 0.3, 0.1)}
    pts = hand(flex)
    pts[4] = pts[8] + np.array([0.004, 0.0, 0.0], dtype=np.float32)
    if R is not None:
        pts = (pts @ R.T).astype(np.float32)
    return pts
