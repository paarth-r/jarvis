"""Decides which single gesture is active, frame by frame.

Each frame the hand shape yields at most one *candidate* (fist, a custom
gesture, or point, in that priority). The arbiter turns candidates into an
*active* gesture: a candidate must lead for its dwell before it starts, and
the active gesture survives short tracking dropouts. Poses the hand only
passes through (the index straightening for a frame on the way into a fist)
never start anything.

Pinch is not arbitrated here: the pinch FSM in gesture_hci owns click
latency. The engine only hears the pinch state: while a pinch is pending
only a fist can take over (and cancels it); while dragging nothing can.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .builtins import is_fist, is_pointing

# shape -> (best custom gesture name or None, {name: match score 0-1})
Matcher = Callable[[np.ndarray], Tuple[Optional[str], Dict[str, float]]]


class GestureArbiter:
    """Pure temporal logic: candidates in, one active gesture out."""

    def __init__(self, dwell_s: float = 0.15, release_s: float = 0.1,
                 dwell: Optional[Dict[str, float]] = None) -> None:
        self._dwell_s = dwell_s
        self._release_s = release_s
        self._dwell = dict(dwell or {})
        self.active: Optional[str] = None
        self._pending: Optional[str] = None
        self._pending_since = 0.0
        self._lost_since: Optional[float] = None

    def dwell_for(self, name: str) -> float:
        return self._dwell.get(name, self._dwell_s)

    def update(self, candidate: Optional[str], now: float) -> Tuple[Optional[str], Optional[str]]:
        """Returns (started, ended) gesture names for this frame."""
        ended = None
        if self.active is not None:
            if candidate == self.active:
                self._lost_since = None
                return None, None
            if self._lost_since is None:
                self._lost_since = now
            if now - self._lost_since < self._release_s:
                return None, None
            ended, self.active, self._lost_since = self.active, None, None
            self._pending = None

        if candidate is None:
            self._pending = None
            return None, ended
        if candidate != self._pending:
            self._pending, self._pending_since = candidate, now
        if now - self._pending_since >= self.dwell_for(candidate):
            self.active, self._pending = candidate, None
            return candidate, ended
        return None, ended

    def reset(self) -> Optional[str]:
        """Drop all state. Returns the gesture that was active, if any."""
        ended = self.active
        self.active, self._pending, self._lost_since = None, None, None
        return ended


@dataclass
class GestureFrame:
    active: Optional[str]                 # the one gesture in effect now
    candidate: Optional[str]              # what this frame's shape looked like
    started: Optional[str] = None         # became active this frame
    ended: Optional[str] = None           # stopped being active this frame
    scores: Dict[str, float] = field(default_factory=dict)  # custom match scores


class GestureEngine:
    def __init__(self, config: Optional[dict] = None, matcher: Optional[Matcher] = None) -> None:
        cfg = config or {}
        self._arbiter = GestureArbiter(
            dwell_s=cfg.get("dwell_s", 0.15),
            release_s=cfg.get("release_s", 0.1),
            dwell={"fist": 0.1, "point": 0.3, **(cfg.get("dwell") or {})},
        )
        self.matcher = matcher

    @property
    def active(self) -> Optional[str]:
        return self._arbiter.active

    def classify(self, shape: Optional[np.ndarray], pinch_state: str = "open"
                 ) -> Tuple[Optional[str], Dict[str, float]]:
        if shape is None or pinch_state == "dragging":
            return None, {}
        if is_fist(shape):
            return "fist", {}
        if pinch_state != "open":
            return None, {}
        name, scores = self.matcher(shape) if self.matcher else (None, {})
        if name is not None:
            return name, scores
        if is_pointing(shape):
            return "point", scores
        return None, scores

    def update(self, shape: Optional[np.ndarray], pinch_state: str, now: float) -> GestureFrame:
        """pinch_state: "open" | "pending" | "dragging" (see PinchFSM.state)."""
        candidate, scores = self.classify(shape, pinch_state)
        started, ended = self._arbiter.update(candidate, now)
        return GestureFrame(self._arbiter.active, candidate, started, ended, scores)

    def reset(self) -> Optional[str]:
        return self._arbiter.reset()
