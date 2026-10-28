"""Fist-tilt scroll: tilting a closed fist away from the angle it started at
scrolls, the further the faster (rate control, like a joystick).

Pitch (knuckles tipping toward the camera) scrolls down the page; tipping
back scrolls up. Roll (turning the forearm) scrolls sideways when enabled.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .features import tilt


class TiltScroller:
    def __init__(self, deadzone_deg: float = 8.0, full_deg: float = 35.0,
                 max_px_s: float = 2500.0, curve: float = 1.5,
                 smoothing: float = 0.4, horizontal: bool = False,
                 invert_y: bool = False, invert_x: bool = False,
                 max_dt_s: float = 0.1) -> None:
        self.deadzone = deadzone_deg       # tilt that does nothing (hand wobble)
        self.full = full_deg               # tilt that reaches max speed
        self.max_px_s = max_px_s
        self.curve = curve                 # >1 = finer control near the dead zone
        self.smoothing = smoothing         # EMA weight of the newest angle (1 = off)
        self.horizontal = horizontal
        self.sign_y = -1.0 if invert_y else 1.0
        self.sign_x = -1.0 if invert_x else 1.0
        self.max_dt = max_dt_s             # cap per-frame time so a stall can't jump
        self._ref: Optional[np.ndarray] = None
        self._last_t = 0.0
        self._angles = np.zeros(2)
        self._carry = np.zeros(2)
        self.pitch = self.roll = 0.0       # smoothed, for the UI gauge

    def start(self, frame: np.ndarray, now: float) -> None:
        self._ref = frame
        self._last_t = now
        self._angles[:] = 0.0
        self._carry[:] = 0.0
        self.pitch = self.roll = 0.0

    def stop(self) -> None:
        self._ref = None

    @property
    def active(self) -> bool:
        return self._ref is not None

    def speed(self, angle_deg: float) -> float:
        """Scroll speed in px/s for a tilt angle."""
        span = max(self.full - self.deadzone, 1e-6)
        x = min(max((abs(angle_deg) - self.deadzone) / span, 0.0), 1.0)
        return float(np.sign(angle_deg) * self.max_px_s * x ** self.curve)

    def update(self, frame: np.ndarray, now: float) -> Tuple[int, int]:
        """Whole pixels to scroll this frame: (dx right, dy down the page)."""
        if self._ref is None:
            return 0, 0
        dt = min(max(now - self._last_t, 0.0), self.max_dt)
        self._last_t = now
        a = self.smoothing
        self._angles = a * np.array(tilt(self._ref, frame)) + (1 - a) * self._angles
        self.pitch, self.roll = float(self._angles[0]), float(self._angles[1])
        vy = self.sign_y * self.speed(self.pitch)
        vx = self.sign_x * self.speed(self.roll) if self.horizontal else 0.0
        self._carry += np.array([vx, vy]) * dt
        whole = np.trunc(self._carry)
        self._carry -= whole
        return int(whole[0]), int(whole[1])
