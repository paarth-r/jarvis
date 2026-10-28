"""Few-shot custom gestures: nearest neighbour against the saved takes.

No training step ($1-recognizer style: a handful of templates per gesture
is enough), so a gesture works the moment it is saved. Each gesture has an
acceptance radius fitted to how much its own takes vary; a hand farther
than that from every gesture is "no gesture" (the job Model Maker's `none`
class does).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def take_samples(takes: Sequence[np.ndarray], per_take: int = 20) -> np.ndarray:
    """Stack takes ((frames, D) each) into templates, evenly subsampled so a
    long take doesn't outweigh a short one."""
    out = []
    for t in takes:
        t = np.asarray(t, dtype=np.float32)
        idx = np.linspace(0, len(t) - 1, min(per_take, len(t))).round().astype(int)
        out.append(t[idx])
    return np.concatenate(out, axis=0)


def cross_take_distances(takes: Sequence[np.ndarray]) -> np.ndarray:
    """For every frame, distance to the nearest frame of a *different* take.
    Frames within one take are near-duplicates, so only other takes say how
    far a fresh performance of the gesture can land."""
    takes = [np.asarray(t, dtype=np.float32) for t in takes]
    out = []
    for i, t in enumerate(takes):
        others = np.concatenate([o for j, o in enumerate(takes) if j != i], axis=0)
        d = np.linalg.norm(t[:, None, :] - others[None, :, :], axis=-1).min(axis=1)
        out.append(d)
    return np.concatenate(out)


def fit_radius(takes: Sequence[np.ndarray], spread: float = 1.5) -> float:
    """Acceptance radius: a margin over the 90th-percentile cross-take
    distance, floored so a suspiciously consistent set isn't impossible to hit."""
    if len(takes) < 2:
        return Matcher.MIN_RADIUS
    return float(max(Matcher.MIN_RADIUS, spread * np.percentile(cross_take_distances(takes), 90)))


class Matcher:
    MIN_RADIUS = 0.25        # feature units (radians / palm widths)
    MARGIN = 0.8             # best must beat the runner-up's relative distance by this

    def __init__(self) -> None:
        self._names: List[str] = []
        self._samples: List[np.ndarray] = []
        self._radii: List[float] = []

    def add(self, name: str, samples: np.ndarray, radius: float) -> None:
        self._names.append(name)
        self._samples.append(np.asarray(samples, dtype=np.float32))
        self._radii.append(float(radius))

    @classmethod
    def from_library(cls, library) -> "Matcher":
        m = cls()
        for name, g in library.gestures.items():
            m.add(name, np.asarray(g["samples"], dtype=np.float32), g["radius"])
        return m

    def __len__(self) -> int:
        return len(self._names)

    def distances(self, f: np.ndarray) -> Dict[str, float]:
        """Distance from feature vector f to each gesture, in units of that
        gesture's radius (< 1 = inside)."""
        return {n: float(np.linalg.norm(s - f, axis=1).min() / r)
                for n, s, r in zip(self._names, self._samples, self._radii)}

    def match(self, f: np.ndarray) -> Tuple[Optional[str], Dict[str, float]]:
        """(best gesture or None, {name: score 0-1}). Score 1 = dead on,
        0 = at or beyond the radius."""
        if not self._names:
            return None, {}
        rel = self.distances(f)
        scores = {n: max(0.0, 1.0 - d) for n, d in rel.items()}
        ranked = sorted(rel.items(), key=lambda kv: kv[1])
        best, d1 = ranked[0]
        if d1 >= 1.0:
            return None, scores
        if len(ranked) > 1 and ranked[1][1] < 1.0 and d1 > self.MARGIN * ranked[1][1]:
            return None, scores                  # two gestures claim this hand
        return best, scores
