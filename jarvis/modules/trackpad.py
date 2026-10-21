from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from .one_euro import OneEuroFilter


def accel_gain(speed: float, gain_min: float, gain_max: float,
               speed_lo: float, speed_hi: float) -> float:
    """Pointer-acceleration transfer function: px of cursor per metre of hand.
    Flat gain_min below speed_lo (precision), flat gain_max above speed_hi
    (reach), smoothstep in between so there is no felt 'gear change'."""
    if speed <= speed_lo:
        return gain_min
    if speed >= speed_hi:
        return gain_max
    u = (speed - speed_lo) / (speed_hi - speed_lo)
    s = u * u * (3.0 - 2.0 * u)
    return gain_min + s * (gain_max - gain_min)


class TrackpadMapper:
    """Pure relative mapper — no hardware dependency. Turns a stream of hand
    positions (metres, camera frame) into integer cursor deltas (pixels),
    like a trackpad: only motion matters, not where the hand is.

    Positions are 1-euro filtered, then each frame's displacement is scaled
    by a speed-dependent gain. Sub-pixel remainders carry over so slow
    motion isn't truncated away. X is mirrored (the camera faces the user).
    """

    def __init__(
        self,
        gain_min: float = 1500.0,    # px per metre at slow speed (~1.5 px/mm)
        gain_max: float = 40000.0,   # px per metre at fast speed (~40 px/mm)
        speed_lo: float = 0.03,      # m/s; at/below -> gain_min
        speed_hi: float = 0.35,      # m/s; at/above -> gain_max
        deadzone_mps: float = 0.005, # m/s; slower than this is treated as still
        min_cutoff: float = 1.0,     # 1-euro: Hz cutoff at rest
        beta: float = 10.0,          # 1-euro: cutoff increase per m/s
        max_gap_s: float = 0.25,     # longer gap (hand lost) restarts tracking
    ) -> None:
        self._gain = (gain_min, gain_max, speed_lo, speed_hi)
        self._deadzone = deadzone_mps
        self._max_gap = max_gap_s
        self._filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.reset()

    def reset(self) -> None:
        """Forget the last position — the next update moves nothing. Used for
        the clutch (hand 'lifted') and when tracking drops out."""
        self._filter.reset()
        self._last: Optional[np.ndarray] = None
        self._last_t: float = 0.0
        self._remainder = np.zeros(2)

    def update(self, pos_xy: np.ndarray, t: float) -> Tuple[int, int]:
        if self._last is not None and t - self._last_t > self._max_gap:
            self.reset()

        p = self._filter(np.asarray(pos_xy, dtype=np.float64)[:2], t)
        if self._last is None:
            self._last, self._last_t = p, t
            return (0, 0)

        dt = t - self._last_t
        if dt <= 0:
            return (0, 0)
        d = p - self._last
        self._last, self._last_t = p, t

        speed = float(np.linalg.norm(d)) / dt
        if speed < self._deadzone:
            return (0, 0)

        move = np.array([-d[0], d[1]]) * accel_gain(speed, *self._gain)
        move += self._remainder
        whole = np.trunc(move)
        self._remainder = move - whole
        return (int(whole[0]), int(whole[1]))
