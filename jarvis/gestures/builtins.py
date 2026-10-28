"""Rule-based built-in hand shapes. They compare distances from the wrist, so
they work on any (21, 3) hand in metres, rotated or not."""
from __future__ import annotations

import numpy as np

WRIST = 0
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

# Reserved built-ins: pinch = click/drag, fist = scroll. point is bindable.
RESERVED = ("pinch", "fist")
BUILTINS = ("pinch", "fist", "point")


def _extended(landmarks_3d: np.ndarray, tip: int, pip: int) -> bool:
    """A finger is extended when its tip is farther from the wrist than its PIP."""
    wrist = landmarks_3d[WRIST]
    return bool(np.linalg.norm(landmarks_3d[tip] - wrist)
                > np.linalg.norm(landmarks_3d[pip] - wrist))


def is_pointing(landmarks_3d: np.ndarray) -> bool:
    """True when the index finger is extended and middle/ring/pinky are curled."""
    index_out = _extended(landmarks_3d, INDEX_TIP, INDEX_PIP)
    others_curled = (
        not _extended(landmarks_3d, MIDDLE_TIP, MIDDLE_PIP)
        and not _extended(landmarks_3d, RING_TIP, RING_PIP)
        and not _extended(landmarks_3d, PINKY_TIP, PINKY_PIP)
    )
    return index_out and others_curled


def is_fist(landmarks_3d: np.ndarray) -> bool:
    """True when all four fingertips are folded back into the palm (each tip
    no farther from the wrist than its own knuckle). Stricter than 'not
    extended' on purpose: a pinch bends the index but leaves its tip well
    beyond the knuckle, so pinching never reads as a fist."""
    wrist = landmarks_3d[WRIST]
    for tip, mcp in [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                     (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)]:
        if (np.linalg.norm(landmarks_3d[tip] - wrist)
                > np.linalg.norm(landmarks_3d[mcp] - wrist)):
            return False
    return True
