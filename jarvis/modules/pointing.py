from __future__ import annotations
import asyncio
from typing import Tuple

import numpy as np

from ..events import WorldPoseEvent, PointEvent
from ..runtime.module_base import Module

from ..gestures.builtins import (  # noqa: F401  (re-exported for callers)
    WRIST, INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    RING_MCP, RING_PIP, RING_TIP, PINKY_MCP, PINKY_PIP, PINKY_TIP,
    is_fist, is_pointing,
)


def compute_ray(landmarks_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pointing ray: origin at the index MCP knuckle, direction toward the tip.
    Returns (origin (3,), unit direction (3,))."""
    origin = landmarks_3d[INDEX_MCP].astype(np.float64)
    d = landmarks_3d[INDEX_TIP].astype(np.float64) - origin
    n = float(np.linalg.norm(d))
    direction = d / n if n > 0 else d
    return origin, direction


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
