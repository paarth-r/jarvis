from __future__ import annotations
from typing import List, Optional, Tuple

import cv2
import numpy as np

# MediaPipe 21-landmark hand connections (thumb, index, middle, ring, pinky, palm)
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]

Point3D = Optional[Tuple[float, float, float]]


def hand_points(world_landmarks: Optional[np.ndarray]) -> List[Point3D]:
    """Map (21,3) world landmarks (X right, Y down, Z fwd) to display tuples
    (right, depth, up) = (x, z, -y). None → list of 21 Nones."""
    if world_landmarks is None:
        return [None] * 21
    pts: List[Point3D] = []
    for i in range(world_landmarks.shape[0]):
        x, y, z = (float(v) for v in world_landmarks[i])
        pts.append((x, z, -y))
    return pts


def draw_landmarks(frame: np.ndarray, lm2d: np.ndarray, connections) -> np.ndarray:
    """Return a BGR copy of frame with normalized (21,3) landmarks drawn.
    Accepts mono (HxW) or BGR (HxWx3) input. Does not mutate the input."""
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        canvas = frame.copy()
    h, w = canvas.shape[:2]
    px = [(int(lm2d[i, 0] * w), int(lm2d[i, 1] * h)) for i in range(lm2d.shape[0])]
    for a, b in connections:
        if a < len(px) and b < len(px):
            cv2.line(canvas, px[a], px[b], (0, 255, 0), 2)
    for pt in px:
        cv2.circle(canvas, pt, 4, (0, 255, 255), -1)
    return canvas
