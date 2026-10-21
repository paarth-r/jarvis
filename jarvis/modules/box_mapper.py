from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from .one_euro import OneEuroFilter


class BoxMapper:
    """Pure absolute mapper — no hardware dependency. A small box of hand
    travel (metres, camera frame) maps linearly onto the whole screen, so
    hand position is cursor position with no acceleration.

    The box comes to the hand instead of the hand hunting for the box:
    - on (re)acquire it is placed so the hand maps to where the cursor
      already is — no jump when the hand enters view;
    - pushing past an edge slides the box along, so the cursor rides the
      screen edge and comes straight back when the hand reverses.

    Positions are 1-euro filtered: rock-steady near rest, low lag when moving.
    X is mirrored (the camera faces the user).
    """

    def __init__(
        self,
        box_width_m: float = 0.15,   # hand travel for the full screen width
        box_height_m: float = 0.10,  # hand travel for the full screen height
        min_cutoff: float = 0.8,     # 1-euro: Hz cutoff at rest
        beta: float = 10.0,          # 1-euro: cutoff increase per m/s
        max_gap_s: float = 0.25,     # longer gap (hand lost) re-anchors the box
    ) -> None:
        self._size = np.array([box_width_m, box_height_m], dtype=np.float64)
        self._max_gap = max_gap_s
        self._filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._center: Optional[np.ndarray] = None
        self._last_t: float = 0.0
        self._out = np.array([0.5, 0.5])   # last cursor position, normalised

    def _offset(self, norm: np.ndarray) -> np.ndarray:
        """Hand offset from box centre (metres) that maps to `norm`."""
        return np.array([0.5 - norm[0], norm[1] - 0.5]) * self._size

    def update(self, pos_xy: np.ndarray, t: float) -> Tuple[float, float]:
        if self._center is not None and t - self._last_t > self._max_gap:
            self._center = None
            self._filter.reset()
        self._last_t = t

        p = self._filter(np.asarray(pos_xy, dtype=np.float64)[:2], t)
        if self._center is None:
            self._center = p - self._offset(self._out)

        # edge push: keep the hand inside the box by dragging the box along
        half = self._size / 2.0
        rel = p - self._center
        over = np.abs(rel) - half
        self._center += np.where(over > 0, np.sign(rel) * over, 0.0)
        rel = p - self._center

        self._out = np.array([0.5 - rel[0] / self._size[0],
                              0.5 + rel[1] / self._size[1]]).clip(0.0, 1.0)
        return (float(self._out[0]), float(self._out[1]))
