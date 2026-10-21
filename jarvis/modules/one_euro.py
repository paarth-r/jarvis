from __future__ import annotations
import math
from typing import Optional

import numpy as np


def _alpha(cutoff_hz: float, dt: float) -> float:
    """Smoothing factor of a first-order low-pass at cutoff_hz for step dt."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """1-euro filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass whose
    cutoff rises with speed — heavy smoothing when the hand is near still
    (kills jitter), light smoothing when it moves fast (kills lag).

    Filters a vector; one cutoff from the vector's speed magnitude so both
    axes are smoothed together (no diagonal distortion).

    Tuning: lower min_cutoff for less jitter at rest, raise beta for less
    lag in fast motion. beta is in units of 1/(signal units), so a signal in
    metres needs a much larger beta than one in pixels.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: float = 0.0

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x.copy()

        dt = t - self._t
        if dt <= 0:
            return self._x.copy()
        self._t = t

        dx = (x - self._x) / dt
        a_d = _alpha(self._d_cutoff, dt)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx

        cutoff = self._min_cutoff + self._beta * float(np.linalg.norm(self._dx))
        a = _alpha(cutoff, dt)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()
