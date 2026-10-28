"""Turn the hand shapes that arrived during a recording window into a frame
list with None where the hand was lost, which is what the quality check
counts as dropouts. Poses only arrive when a hand is seen, so lost time is
read from gaps in their timestamps."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DT = 1 / 30
# Gaps up to this many frame intervals are scheduling hiccups (the pose model
# skipping a frame under load), not the tracker losing the hand.
LOST_AFTER = 3.0


def to_frames(samples: Sequence[Tuple[float, np.ndarray]], start: float, end: float,
              ) -> List[Optional[np.ndarray]]:
    """samples: (timestamp, hand_shape) in time order. The frame interval is
    the median gap between samples, so a pose model running at half the
    camera rate (backpressure) is not mistaken for a lost hand."""
    if not samples:
        return [None] * max(0, round((end - start) / DEFAULT_DT))
    ts = np.array([t for t, _ in samples])
    dt = max(float(np.median(np.diff(ts))) if len(ts) > 1 else DEFAULT_DT, 1e-3)

    def lost(span: float) -> int:
        """Frames lost in a stretch of `span` seconds without a pose."""
        return round(span / dt) if span > LOST_AFTER * dt else 0

    # Frame slots run start, start + dt, ..., end - dt. Before the first pose
    # and between poses, a gap of n intervals hides n - 1 frames (the first
    # slot sits at `start` itself, so the head hides all n).
    frames: List[Optional[np.ndarray]] = [None] * lost(ts[0] - start)
    prev = ts[0]
    for t, shape in samples:
        frames += [None] * max(0, lost(t - prev) - 1)
        frames.append(shape)
        prev = t
    frames += [None] * max(0, lost(end - prev) - 1)
    return frames
