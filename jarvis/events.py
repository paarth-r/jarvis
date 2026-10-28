from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FrameEvent:
    camera_id: str
    frame: np.ndarray       # HxW uint8 mono (OV9281 native)
    timestamp: float        # time.perf_counter() seconds


@dataclass
class PoseEvent:
    camera_id: str
    landmarks_2d: np.ndarray  # (21, 3) normalized [x∈0-1, y∈0-1, z_rel]
    timestamp: float
    hand_side: str            # "left" | "right"
    # (21, 3) MediaPipe hand_world_landmarks: metres, hand-centred, camera-
    # oriented, with model-estimated depth. None when the model gave none.
    world_landmarks: Optional[np.ndarray] = None


@dataclass
class WorldPoseEvent:
    # WorldPoseEvent is the stereo boundary.
    # Everything upstream is camera-aware; everything downstream is not.
    landmarks_3d: np.ndarray  # (21, 3) meters in world coordinate frame
    timestamp: float
    confidence: float         # 1.0 = full stereo, 0.5 = mono fallback
    hand_side: str
    # (21, 3) hand shape in metres (see PoseEvent.world_landmarks). Use this,
    # not landmarks_3d, for shape and orientation: mono projection flattens z.
    hand_shape: Optional[np.ndarray] = None


@dataclass
class IntentEvent:
    type: str        # "click" | "press" | "release" | "idle" | "move" | "gesture" | "point" | "command"
    confidence: float
    payload: dict
    timestamp: float


@dataclass
class GestureEvent:
    name: str        # "fist" | "point" | a custom gesture's name
    phase: str       # "start" | "end"
    timestamp: float
    payload: dict = field(default_factory=dict)


@dataclass
class StudioModeEvent:
    # The gesture studio is recording takes: cursor control and gesture
    # actions pause so making the gesture doesn't click or fire anything.
    recording: bool


@dataclass
class PointEvent:
    world_pos: np.ndarray   # ray origin in world coords (3,)
    direction: np.ndarray   # unit vector (3,)
    target: Optional[str]   # resolved target label, if known
    confidence: float
    timestamp: float


@dataclass
class ActionEvent:
    action_type: str
    params: dict
    timestamp: float


@dataclass
class SystemEvent:
    type: str        # "module_started" | "module_stopped" | "module_error"
    module_id: str
    payload: dict


@dataclass
class GazeEvent:
    norm_x: float            # 0-1 across the calibrated display, smoothed
    norm_y: float            # 0-1 down the calibrated display, smoothed
    timestamp: float
    confidence: float        # 1.0 = face found this frame
