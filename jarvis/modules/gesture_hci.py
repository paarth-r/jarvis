from __future__ import annotations
import asyncio
from enum import Enum
from typing import Optional

import numpy as np

from ..events import GestureEvent, WorldPoseEvent, IntentEvent
from ..gestures.builtins import is_fist
from ..gestures.engine import GestureEngine, GestureFrame
from ..runtime.module_base import Module
from .trackpad import TrackpadMapper
from .box_mapper import BoxMapper

# MediaPipe hand landmark indices
INDEX_MCP = 5
THUMB_MCP = 2
INDEX_TIP = 8
THUMB_TIP = 4


class _PinchState(Enum):
    OPEN = "open"
    PENDING = "pending"    # pinched, waiting to see if it's a click or a drag
    DRAGGING = "dragging"
    LOCKED = "locked"      # another gesture owns the hand; re-arms when the thumb opens


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

    @property
    def is_pending(self) -> bool:
        """Pinched, not yet a drag — the cursor should hold still."""
        return self._state == _PinchState.PENDING

    @property
    def state(self) -> str:
        """"open" | "pending" | "dragging" | "locked"."""
        return self._state.value

    def cancel(self) -> str:
        """Abandon the current pinch without clicking. Returns 'release' if a
        drag was in progress (so the button gets let go), else 'idle'."""
        was_dragging = self._state == _PinchState.DRAGGING
        self._state = _PinchState.OPEN
        return "release" if was_dragging else "idle"

    def lock(self) -> str:
        """Like cancel, but no new pinch can start until the thumb has opened
        past release_thresh once: a thumb resting on the index as a fist
        opens must not click."""
        result = self.cancel()
        self._state = _PinchState.LOCKED
        return result

    def update(self, distance: float, now: float) -> str:
        if self._state == _PinchState.LOCKED:
            if distance > self._release:
                self._state = _PinchState.OPEN
            return "idle"

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
        # "absolute": hand position = screen position (screen_gain).
        # "box": hand position = screen position from a small self-placing
        #        box of travel, 1-euro filtered (see BoxMapper).
        # "trackpad": hand motion moves the cursor, with acceleration; a fist
        # is the clutch (lift your finger off the pad).
        self._mode: str = self.config.get("mode", "absolute")
        # Import MouseController lazily so Quartz import failure doesn't break tests
        from mouse_control.control import MouseController
        if self._mode == "box":
            # BoxMapper owns gain and smoothing; the controller passes through
            self._controller = MouseController(
                move_gain=1.0, smoothing=1.0,
                move_threshold_px=self.config.get("move_threshold_px", 1.0),
            )
        else:
            self._controller = MouseController(
                smoothing=self.config.get("smoothing", 0.3),
                move_threshold_px=self.config.get("move_threshold_px", 2.0),
            )
        self._gain: float = self.config.get("screen_gain", 2.5)
        self._box = BoxMapper(**(self.config.get("box", {}) or {}))
        self._frozen: Optional[tuple] = None
        self._prev_norm: Optional[tuple] = None
        tp = self.config.get("trackpad", {}) or {}
        self._trackpad = TrackpadMapper(**tp)
        self._clutched = False
        # One gesture at a time (see jarvis/gestures/engine.py). Fist holds
        # the cursor; any active gesture locks out pinch.
        self.engine = GestureEngine(self.config.get("gestures", {}) or {})
        self.last_frame: Optional[GestureFrame] = None
        self.bus.subscribe(WorldPoseEvent, self._on_pose)

    async def _on_pose(self, event: WorldPoseEvent) -> None:
        hold = await self._arbitrate(event)
        if self._mode == "trackpad":
            await self._on_pose_trackpad(event, hold)
        elif self._mode == "box":
            await self._on_pose_box(event, hold)
        else:
            await self._on_pose_absolute(event, hold)

    async def _arbitrate(self, event: WorldPoseEvent) -> bool:
        """Run the gesture engine, publish start/end, lock pinch out while a
        gesture owns the hand. Returns True when the cursor should hold still
        (a fist, from its first frame, before its dwell completes)."""
        pinch_state = self._fsm.state
        frame = self.engine.update(
            event.hand_shape, "open" if pinch_state == "locked" else pinch_state,
            event.timestamp,
        )
        self.last_frame = frame
        if frame.ended:
            await self.bus.publish(GestureEvent(frame.ended, "end", event.timestamp))
        if frame.started:
            await self.bus.publish(GestureEvent(frame.started, "start", event.timestamp))
        fist = frame.active == "fist" or frame.candidate == "fist"
        if (fist or frame.active is not None) and self._fsm.state != "dragging":
            self._fsm.lock()
        return fist

    async def _on_pose_box(self, event: WorldPoseEvent, hold: bool = False) -> None:
        lm = event.landmarks_3d
        mid = (lm[INDEX_MCP, :2] + lm[THUMB_MCP, :2]) / 2.0
        norm_x, norm_y = self._box.update(mid, event.timestamp)

        dist = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]))
        intent = self._fsm.update(dist, event.timestamp)

        # Pinching shifts the knuckles; hold the cursor where it was the frame
        # before the fingers closed (the closing frame already carries some
        # shift), then re-anchor the box there so it doesn't jump when the
        # pinch resolves into a click or a drag. A fist holds it the same way.
        if self._fsm.is_pending or hold:
            if self._frozen is None:
                self._frozen = self._prev_norm or (norm_x, norm_y)
            norm_x, norm_y = self._frozen
        elif self._frozen is not None:
            norm_x, norm_y = self._frozen
            self._box.anchor(self._frozen)
            self._frozen = None
        self._prev_norm = (norm_x, norm_y)

        if intent == "click":
            self._controller.click(norm_x, norm_y)
        elif intent == "press":
            self._controller.mouse_down(norm_x, norm_y)
        elif intent == "release":
            self._controller.mouse_up(norm_x, norm_y)
        else:
            self._controller.move(norm_x, norm_y)

        await self.bus.publish(IntentEvent(
            type=intent,
            confidence=event.confidence,
            payload={"norm_x": norm_x, "norm_y": norm_y, "dist_m": dist},
            timestamp=event.timestamp,
        ))

    async def _on_pose_trackpad(self, event: WorldPoseEvent, hold: bool = False) -> None:
        lm = event.landmarks_3d
        dx = dy = 0
        dist = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]))

        self._clutched = is_fist(lm) or hold
        if self._clutched:
            # Hand 'lifted': forget position so there's no jump on re-entry,
            # and drop any pinch (a closing fist can brush thumb to index).
            self._trackpad.reset()
            intent = self._fsm.cancel()
        else:
            mid = (lm[INDEX_MCP, :2] + lm[THUMB_MCP, :2]) / 2.0
            dx, dy = self._trackpad.update(mid, event.timestamp)
            intent = self._fsm.update(dist, event.timestamp)
            if self._fsm.is_pending:
                # Pinching shifts the knuckles; freeze so the click lands
                # where the cursor was when the fingers closed.
                dx = dy = 0

        # Button events before the move: a click/release fires at the spot
        # the cursor was, not after the opening fingers nudge it.
        if intent == "click":
            self._controller.click_here()
        elif intent == "press":
            self._controller.mouse_down_here()
        elif intent == "release":
            self._controller.mouse_up_here()
        if dx or dy:
            self._controller.move_by(dx, dy)

        await self.bus.publish(IntentEvent(
            type=intent,
            confidence=event.confidence,
            payload={"dx": dx, "dy": dy, "clutched": self._clutched, "dist_m": dist},
            timestamp=event.timestamp,
        ))

    async def _on_pose_absolute(self, event: WorldPoseEvent, hold: bool = False) -> None:
        lm = event.landmarks_3d

        # Cursor follows midpoint of index and thumb knuckles (MCPs), mirrored on X
        mid_x = (lm[INDEX_MCP, 0] + lm[THUMB_MCP, 0]) / 2.0
        mid_y = (lm[INDEX_MCP, 1] + lm[THUMB_MCP, 1]) / 2.0
        norm_x = max(0.0, min(1.0, 1.0 - (0.5 + self._gain * mid_x)))
        norm_y = max(0.0, min(1.0, 0.5 + self._gain * mid_y))

        if not hold:
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
