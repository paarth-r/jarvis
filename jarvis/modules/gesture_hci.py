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
    PENDING = "pending"    # pinched, waiting to see if it's a click or a drag
    DRAGGING = "dragging"


class PinchFSM:
    """Pure FSM — no hardware dependency. A pinch released within hold_delay
    is a 'click'; one held longer emits 'press' (drag start) and 'release'
    (drag end) when it opens."""

    def __init__(
        self,
        pinch_thresh: float = 0.025,   # meters; enter pinch
        release_thresh: float = 0.045,  # meters; exit pinch (hysteresis)
        hold_delay: float = 0.2,        # seconds pinched before drag starts
    ) -> None:
        self._thresh = pinch_thresh
        self._release = release_thresh
        self._hold_delay = hold_delay
        self._state = _PinchState.OPEN
        self._pinch_start_ts: float = 0.0

    def update(self, distance: float, now: float) -> str:
        if self._state == _PinchState.OPEN:
            if distance < self._thresh:
                self._state = _PinchState.PENDING
                self._pinch_start_ts = now
            return "idle"

        if self._state == _PinchState.PENDING:
            if distance > self._release:
                self._state = _PinchState.OPEN
                return "click"
            if now - self._pinch_start_ts >= self._hold_delay:
                self._state = _PinchState.DRAGGING
                return "press"
            return "idle"

        if self._state == _PinchState.DRAGGING:
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
            hold_delay=self.config.get("drag_hold_s", 0.2),
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
        intent = self._fsm.update(dist, event.timestamp)

        # Quick pinch = click; pinch held past drag_hold_s = button down,
        # movement while held drags (MouseController.move emits drag events
        # while down), open releases.
        if intent == "click":
            self._controller.click(norm_x, norm_y)
        elif intent == "press":
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
