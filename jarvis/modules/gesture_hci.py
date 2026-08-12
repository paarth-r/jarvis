from __future__ import annotations
import asyncio
from enum import Enum
from typing import Optional

import numpy as np

from ..events import WorldPoseEvent, IntentEvent
from ..runtime.module_base import Module

# MediaPipe hand landmark indices
INDEX_MCP = 5
THUMB_MCP = 2
INDEX_TIP = 8
THUMB_TIP = 4


class _PinchState(Enum):
    OPEN = "open"
    HELD = "held"


class PinchFSM:
    """Pure FSM — no hardware dependency. Returns 'press' on OPEN→HELD and
    'release' on HELD→OPEN, enabling click-and-drag."""

    def __init__(
        self,
        pinch_thresh: float = 0.025,   # meters; enter held
        release_thresh: float = 0.045,  # meters; exit held (hysteresis)
    ) -> None:
        self._thresh = pinch_thresh
        self._release = release_thresh
        self._state = _PinchState.OPEN

    def update(self, distance: float) -> str:
        if self._state == _PinchState.OPEN:
            if distance < self._thresh:
                self._state = _PinchState.HELD
                return "press"
            return "idle"

        if self._state == _PinchState.HELD:
            if distance > self._release:
                self._state = _PinchState.OPEN
                return "release"
            return "idle"

        return "idle"


class GestureHCIModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._fsm = PinchFSM(
            pinch_thresh=self.config.get("pinch_thresh_m", 0.025),
            release_thresh=self.config.get("release_thresh_m", 0.045),
        )
        # Import MouseController lazily so Quartz import failure doesn't break tests
        from mouse_control.control import MouseController
        self._controller = MouseController(
            smoothing=self.config.get("smoothing", 0.3),
            move_threshold_px=self.config.get("move_threshold_px", 2.0),
        )
        self._gain: float = self.config.get("screen_gain", 2.5)
        self.bus.subscribe(WorldPoseEvent, self._on_pose)

    async def _on_pose(self, event: WorldPoseEvent) -> None:
        lm = event.landmarks_3d

        # Cursor follows midpoint of index and thumb knuckles (MCPs), mirrored on X
        mid_x = (lm[INDEX_MCP, 0] + lm[THUMB_MCP, 0]) / 2.0
        mid_y = (lm[INDEX_MCP, 1] + lm[THUMB_MCP, 1]) / 2.0
        norm_x = max(0.0, min(1.0, 1.0 - (0.5 + self._gain * mid_x)))
        norm_y = max(0.0, min(1.0, 0.5 + self._gain * mid_y))

        self._controller.move(norm_x, norm_y)

        dist = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]))
        intent = self._fsm.update(dist)

        # Pinch = button held: close presses, movement while held drags
        # (MouseController.move emits drag events while down), open releases.
        if intent == "press":
            self._controller.mouse_down(norm_x, norm_y)
        elif intent == "release":
            self._controller.mouse_up(norm_x, norm_y)

        await self.bus.publish(IntentEvent(
            type=intent,
            confidence=event.confidence,
            payload={"norm_x": norm_x, "norm_y": norm_y, "dist_m": dist},
            timestamp=event.timestamp,
        ))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)
