from __future__ import annotations
import asyncio
import math
from typing import Optional, Tuple

import numpy as np

from ..events import GazeEvent, IntentEvent, WorldPoseEvent
from ..runtime.module_base import Module
from .gesture_hci import PinchFSM

THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 4, 8, 12


class PinchPair:
    """Two pinch FSMs sharing one thumb: thumb-index = left button (quick
    pinch clicks, held pinch drags), thumb-middle = right click.

    The finger nearer the thumb claims the pinch and holds it until it
    reopens; the other reads as wide open meanwhile, so rolling from one
    finger to the other mid-pinch can't fire both buttons."""

    def __init__(self, pinch_thresh: float, release_thresh: float, drag_hold_s: float) -> None:
        self.left = PinchFSM(pinch_thresh, release_thresh, drag_hold_s)
        self.right = PinchFSM(pinch_thresh, release_thresh, hold_delay=math.inf)
        self._thresh = pinch_thresh
        self._owner: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._owner is not None

    def update(self, d_index: float, d_middle: float, now: float) -> Tuple[str, str]:
        """Returns (button, intent): button 'left'|'right'|'', intent as PinchFSM."""
        if self._owner is None:
            if min(d_index, d_middle) >= self._thresh:
                return "", "idle"
            self._owner = "left" if d_index <= d_middle else "right"
        fsm = self.left if self._owner == "left" else self.right
        dist = d_index if self._owner == "left" else d_middle
        intent = fsm.update(dist, now)
        button = self._owner
        if intent in ("click", "release"):
            self._owner = None
        return button, intent


class _QuartzPointer:
    """Absolute cursor + buttons on the main display, in Quartz points."""

    def __init__(self) -> None:
        from Quartz.CoreGraphics import CGDisplayBounds, CGMainDisplayID
        b = CGDisplayBounds(CGMainDisplayID())
        self.origin = (b.origin.x, b.origin.y)
        self.size = (b.size.width, b.size.height)

    def to_points(self, nx: float, ny: float) -> Tuple[float, float]:
        return (self.origin[0] + nx * (self.size[0] - 1),
                self.origin[1] + ny * (self.size[1] - 1))

    def _post(self, kind: str, x: float, y: float) -> None:
        from Quartz import CoreGraphics as Q
        etype, button = {
            "move": (Q.kCGEventMouseMoved, Q.kCGMouseButtonLeft),
            "drag": (Q.kCGEventLeftMouseDragged, Q.kCGMouseButtonLeft),
            "left_down": (Q.kCGEventLeftMouseDown, Q.kCGMouseButtonLeft),
            "left_up": (Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft),
            "right_down": (Q.kCGEventRightMouseDown, Q.kCGMouseButtonRight),
            "right_up": (Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight),
        }[kind]
        Q.CGEventPost(Q.kCGHIDEventTap, Q.CGEventCreateMouseEvent(None, etype, (x, y), button))

    def post(self, kind: str, nx: float, ny: float) -> None:
        self._post(kind, *self.to_points(nx, ny))


class GazeHCIModule(Module):
    """Eyes point, fingers click.

    follow (default): the cursor follows the smoothed gaze. jump_on_pinch:
    the cursor stays put until a pinch, then jumps to the gaze point and
    clicks there. Either way the click lands at the halo's centre (the gaze
    estimate) - snapping to the nearest control is a later step.
    """

    def __init__(self, module_id: str, config: dict, pointer=None) -> None:
        super().__init__(module_id, config)
        self._pointer = pointer

    async def setup(self, bus) -> None:
        await super().setup(bus)
        if self._pointer is None:
            self._pointer = _QuartzPointer()
        self._jump_on_pinch: bool = bool(self.config.get("jump_on_pinch", False))
        self._move_threshold_px: float = self.config.get("move_threshold_px", 3.0)
        self._pinch = PinchPair(
            pinch_thresh=self.config.get("pinch_thresh_m", 0.025),
            release_thresh=self.config.get("release_thresh_m", 0.045),
            drag_hold_s=self.config.get("drag_hold_s", 0.3),
        )
        self._gaze: Optional[Tuple[float, float]] = None
        self._cursor: Optional[Tuple[float, float]] = None   # last posted, normalized
        self._left_down = False
        self.bus.subscribe(GazeEvent, self._on_gaze)
        self.bus.subscribe(WorldPoseEvent, self._on_pose)

    def _moved_enough(self, p: Tuple[float, float]) -> bool:
        if self._cursor is None:
            return True
        w, h = self._pointer.size
        return math.hypot((p[0] - self._cursor[0]) * w, (p[1] - self._cursor[1]) * h) >= self._move_threshold_px

    def _move(self, p: Tuple[float, float]) -> None:
        self._pointer.post("drag" if self._left_down else "move", *p)
        self._cursor = p

    async def _on_gaze(self, event: GazeEvent) -> None:
        self._gaze = (event.norm_x, event.norm_y)
        if self._jump_on_pinch and not self._left_down:
            return
        # Hold still while a click is forming so it lands where it started;
        # a drag (button held) does follow the eyes.
        if self._pinch.active and not self._left_down:
            return
        if self._moved_enough(self._gaze):
            self._move(self._gaze)

    async def _on_pose(self, event: WorldPoseEvent) -> None:
        lm = event.landmarks_3d
        d_index = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]))
        d_middle = float(np.linalg.norm(lm[THUMB_TIP] - lm[MIDDLE_TIP]))
        button, intent = self._pinch.update(d_index, d_middle, event.timestamp)

        if intent in ("click", "press") and self._gaze is not None:
            target = self._gaze if (self._jump_on_pinch or self._cursor is None) else self._cursor
            if target != self._cursor:
                self._move(target)
        at = self._cursor
        if at is not None:
            if intent == "click":
                self._pointer.post(f"{button}_down", *at)
                self._pointer.post(f"{button}_up", *at)
            elif intent == "press":
                self._left_down = True
                self._pointer.post("left_down", *at)
            elif intent == "release":
                self._left_down = False
                self._pointer.post("left_up", *at)

        await self.bus.publish(IntentEvent(
            type=intent,
            confidence=event.confidence,
            payload={"button": button, "gaze": self._gaze, "d_index_m": d_index, "d_middle_m": d_middle},
            timestamp=event.timestamp,
        ))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)
