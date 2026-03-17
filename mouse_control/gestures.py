"""
Temporal gesture FSMs (no ML).
Cursor at index–thumb midpoint; index–thumb pinch → left click;
thumb–middle pinch → right click.
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


PINCH_CLOSE_THRESH = 0.06
PINCH_OPEN_THRESH = 0.09

RIGHT_PINCH_CLOSE_THRESH = 0.06
RIGHT_PINCH_OPEN_THRESH = 0.09


class PinchState(Enum):
    NONE = "none"
    LEFT_HOLD = "left_pinch_hold"
    RIGHT_HOLD = "right_pinch_hold"


class GestureFSM:
    def __init__(self):
        self._pinch_state = PinchState.NONE

    def _is_left_pinch(self, pose: HandPose) -> bool:
        return pose.thumb_index_dist < PINCH_CLOSE_THRESH

    def _is_left_pinch_released(self, pose: HandPose) -> bool:
        return pose.thumb_index_dist > PINCH_OPEN_THRESH

    def _is_right_pinch(self, pose: HandPose) -> bool:
        return pose.thumb_middle_dist < RIGHT_PINCH_CLOSE_THRESH

    def _is_right_pinch_released(self, pose: HandPose) -> bool:
        return pose.thumb_middle_dist > RIGHT_PINCH_OPEN_THRESH

    def update(self, pose: Optional[HandPose], dt_sec: float) -> GestureIntent:
        if pose is None:
            self._pinch_state = PinchState.NONE
            return GestureIntent(type="idle", confidence=0.0)

        if self._pinch_state == PinchState.NONE:
            if self._is_left_pinch(pose):
                self._pinch_state = PinchState.LEFT_HOLD
                return GestureIntent(type="click", confidence=1.0)
            if self._is_right_pinch(pose):
                self._pinch_state = PinchState.RIGHT_HOLD
                return GestureIntent(type="right_click", confidence=1.0)
            return GestureIntent(type="idle", confidence=1.0)

        if self._pinch_state == PinchState.LEFT_HOLD:
            if self._is_left_pinch_released(pose):
                self._pinch_state = PinchState.NONE
            return GestureIntent(type="idle", confidence=1.0)

        if self._pinch_state == PinchState.RIGHT_HOLD:
            if self._is_right_pinch_released(pose):
                self._pinch_state = PinchState.NONE
            return GestureIntent(type="idle", confidence=1.0)

        return GestureIntent(type="idle", confidence=0.0)

    @property
    def pinch_state(self) -> str:
        return self._pinch_state.value
