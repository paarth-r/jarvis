from __future__ import annotations
import asyncio
from typing import Tuple

import numpy as np

from ..events import WorldPoseEvent, PointEvent
from ..runtime.module_base import Module

# MediaPipe landmark indices
WRIST = 0
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20


def compute_ray(landmarks_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pointing ray: origin at the index MCP knuckle, direction toward the tip.
    Returns (origin (3,), unit direction (3,))."""
    origin = landmarks_3d[INDEX_MCP].astype(np.float64)
    d = landmarks_3d[INDEX_TIP].astype(np.float64) - origin
    n = float(np.linalg.norm(d))
    direction = d / n if n > 0 else d
    return origin, direction


def _extended(landmarks_3d: np.ndarray, tip: int, pip: int) -> bool:
    """A finger is extended when its tip is farther from the wrist than its PIP."""
    wrist = landmarks_3d[WRIST]
    return (np.linalg.norm(landmarks_3d[tip] - wrist)
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


class PointingModule(Module):
    """Emits a PointEvent (world-space ray) while the hand holds a pointing pose."""

    async def setup(self, bus) -> None:
        await super().setup(bus)
        self.bus.subscribe(WorldPoseEvent, self._on_world)

    async def _on_world(self, event: WorldPoseEvent) -> None:
        lm = event.landmarks_3d
        if not is_pointing(lm):
            return
        origin, direction = compute_ray(lm)
        await self.bus.publish(PointEvent(
            world_pos=origin,
            direction=direction,
            target=None,
            confidence=event.confidence,
            timestamp=event.timestamp,
        ))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)
