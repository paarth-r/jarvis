"""
Temporal gesture FSMs (no ML).
Pinch → single click. Cursor always follows hand (mouse move). No drag, no delay.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from hand_pose import HandPose


@dataclass
class GestureIntent:
    type: str
    value: Optional[float] = None
    confidence: float = 0.0


# Pinch: thumb + pinky close → one click. Hysteresis to avoid bounce.
PINCH_CLOSE_THRESH = 0.07
PINCH_OPEN_THRESH = 0.10


class PinchState(Enum):
    NONE = "none"
    HOLD = "pinch_hold"


class GestureFSM:
    def __init__(self):
        self._pinch_state = PinchState.NONE

    def _is_pinch(self, pose: HandPose) -> bool:
        return pose.thumb_pinky_dist < PINCH_CLOSE_THRESH

    def _is_pinch_released(self, pose: HandPose) -> bool:
        return pose.thumb_pinky_dist > PINCH_OPEN_THRESH

    def update(self, pose: Optional[HandPose], dt_sec: float) -> GestureIntent:
        if pose is None:
            self._pinch_state = PinchState.NONE
            return GestureIntent(type="idle", confidence=0.0)

        if self._pinch_state == PinchState.NONE:
            if self._is_pinch(pose):
                self._pinch_state = PinchState.HOLD
                return GestureIntent(type="click", confidence=1.0)
            return GestureIntent(type="idle", confidence=1.0)

        if self._pinch_state == PinchState.HOLD:
            if self._is_pinch_released(pose):
                self._pinch_state = PinchState.NONE
            return GestureIntent(type="idle", confidence=1.0)

        return GestureIntent(type="idle", confidence=0.0)

    @property
    def pinch_state(self) -> str:
        return self._pinch_state.value
