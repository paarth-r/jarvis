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


@dataclass
class WorldPoseEvent:
    # WorldPoseEvent is the stereo boundary.
    # Everything upstream is camera-aware; everything downstream is not.
    landmarks_3d: np.ndarray  # (21, 3) meters in world coordinate frame
    timestamp: float
    confidence: float         # 1.0 = full stereo, 0.5 = mono fallback
    hand_side: str


@dataclass
class IntentEvent:
    type: str        # "click" | "move" | "gesture" | "point" | "command"
    confidence: float
    payload: dict
    timestamp: float


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
