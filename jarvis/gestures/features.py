"""Hand-shape features and palm orientation from a (21, 3) MediaPipe hand.

Features are joint angles and palm-normalized distances, so they don't change
when the hand moves, turns, or comes closer; and they are unsigned, so left and
right hands match. Feed them `WorldPoseEvent.hand_shape`, not `landmarks_3d`
(mono projection flattens depth, which flattens the angles too).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

WRIST = 0
# Chains from the wrist out to the tip; flexion is measured at the 3 inner joints.
_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
_JOINTS = {"thumb": ("cmc", "mcp", "ip")}
_FINGER_JOINTS = ("mcp", "pip", "dip")
# Adjacent first bones whose angle is the "spread" between two fingers.
_SPREADS = (("thumb", (1, 2), (5, 6)), ("index", (5, 6), (9, 10)),
            ("middle", (9, 10), (13, 14)), ("ring", (13, 14), (17, 18)))
_TIPS = (8, 12, 16, 20)
_EPS = 1e-9

FEATURE_NAMES: List[str] = (
    [f"{f}_{j}" for f in _CHAINS for j in _JOINTS.get(f, _FINGER_JOINTS)]
    + [f"spread_{a}" for a, _, _ in _SPREADS]
    + [f"thumb_to_{f}" for f in ("index", "middle", "ring", "pinky")]
)
FEATURE_DIM = len(FEATURE_NAMES)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + _EPS)


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.arccos(np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0)))


def features(shape: np.ndarray) -> np.ndarray:
    """(FEATURE_DIM,) float32: 15 flexion angles (radians), 4 spreads
    (radians), 4 thumb-tip-to-fingertip distances in palm widths."""
    p = np.asarray(shape, dtype=np.float64)
    out: List[float] = []
    for chain in _CHAINS.values():
        bones = np.diff(p[list(chain)], axis=0)
        out += [_angle(bones[k], bones[k + 1]) for k in range(3)]
    for _, (a0, a1), (b0, b1) in _SPREADS:
        out.append(_angle(p[a1] - p[a0], p[b1] - p[b0]))
    palm = np.linalg.norm(p[5] - p[17]) + _EPS
    out += [float(np.linalg.norm(p[4] - p[t]) / palm) for t in _TIPS]
    return np.nan_to_num(np.array(out, dtype=np.float32))


def palm_frame(shape: np.ndarray) -> np.ndarray:
    """3x3 rotation whose columns are the palm's axes in camera coordinates:
    across the knuckles (pinky -> index), along the fingers (wrist ->
    knuckles), and the palm normal. The knuckles stay put in a fist, so this
    tracks a closed hand as well as an open one."""
    p = np.asarray(shape, dtype=np.float64)
    along = _unit(p[[5, 9, 13, 17]].mean(axis=0) - p[WRIST])
    across = p[5] - p[17]
    across = _unit(across - np.dot(across, along) * along)
    normal = np.cross(across, along)
    return np.stack([across, along, normal], axis=1)


def tilt(ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float]:
    """(pitch, roll) in degrees of palm frame `cur` relative to `ref`.
    pitch: rotation about the across-knuckles axis (knuckles tipping
    toward / away from the camera); roll: rotation about the fingers' axis
    (turning the forearm)."""
    m = ref.T @ cur
    along, across = m[:, 1], m[:, 0]
    pitch = np.degrees(np.arctan2(along[2], along[1]))
    roll = np.degrees(np.arctan2(across[2], across[0]))
    return float(pitch), float(roll)
