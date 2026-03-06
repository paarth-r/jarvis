"""
MediaPipe inference + landmark smoothing.
Tracks one dominant hand; outputs normalized landmarks, key distances, velocity (EMA).
"""
import os
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional


# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2   # thumb knuckle (first joint from palm)
THUMB_TIP = 4
INDEX_MCP = 5   # index knuckle (where finger meets palm)
INDEX_TIP = 8
MIDDLE_TIP = 12
PINKY_TIP = 20
INDEX_PIP = 6
MIDDLE_PIP = 10

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


@dataclass
class HandPose:
    """Per-frame hand pose output."""
    landmarks: np.ndarray  # (21, 3) normalized x,y,z
    thumb_index_dist: float
    thumb_pinky_dist: float  # for thumb+pinky pinch (click)
    palm_size: float
    velocity_xy: np.ndarray  # (2,) smoothed velocity in normalized coords
    handedness: str  # "Left" or "Right"


def _ema_update(prev: np.ndarray, curr: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None or prev.size == 0:
        return np.asarray(curr, dtype=np.float64)
    return alpha * np.asarray(curr, dtype=np.float64) + (1 - alpha) * prev


class HandPoseEstimator:
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        landmark_smooth_alpha: float = 0.4,
        velocity_alpha: float = 0.3,
    ):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._landmark_alpha = landmark_smooth_alpha
        self._velocity_alpha = velocity_alpha
        self._smoothed_landmarks: Optional[np.ndarray] = None
        self._prev_landmarks: Optional[np.ndarray] = None
        self._prev_ts: Optional[float] = None
        self._prev_velocity: Optional[np.ndarray] = None
        self._frame_idx = 0

    def process(self, rgb_frame: np.ndarray, timestamp: float) -> Optional[HandPose]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(self._frame_idx * (1000 / 30))
        self._frame_idx += 1
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not results.hand_landmarks:
            self._prev_landmarks = None
            return None
        lm_list = results.hand_landmarks[0]
        handedness = (
            results.handedness[0][0].category_name
            if results.handedness
            else "Unknown"
        )
        if handedness != "Left":
            self.reset()
            return None
        landmarks = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float64)

        # Exponential smoothing on landmarks
        self._smoothed_landmarks = _ema_update(
            self._smoothed_landmarks, landmarks, self._landmark_alpha
        )
        landmarks = self._smoothed_landmarks

        # Key distances (normalized 0–1 space)
        thumb_index_dist = float(
            np.linalg.norm(landmarks[THUMB_TIP, :2] - landmarks[INDEX_TIP, :2])
        )
        thumb_pinky_dist = float(
            np.linalg.norm(landmarks[THUMB_TIP, :2] - landmarks[PINKY_TIP, :2])
        )
        palm_size = float(
            np.linalg.norm(landmarks[WRIST, :2] - landmarks[MIDDLE_TIP, :2])
        )

        # Velocity from landmark motion (use index tip for cursor-like motion)
        velocity_xy = np.zeros(2, dtype=np.float64)
        if self._prev_landmarks is not None and self._prev_ts is not None:
            dt = max(timestamp - self._prev_ts, 1e-6)
            raw_velocity = (landmarks[INDEX_TIP, :2] - self._prev_landmarks[INDEX_TIP, :2]) / dt
            if self._prev_velocity is not None:
                velocity_xy = _ema_update(
                    self._prev_velocity, raw_velocity, self._velocity_alpha
                )
            else:
                velocity_xy = raw_velocity
        self._prev_velocity = velocity_xy
        self._prev_landmarks = landmarks.copy()
        self._prev_ts = timestamp

        return HandPose(
            landmarks=landmarks,
            thumb_index_dist=thumb_index_dist,
            thumb_pinky_dist=thumb_pinky_dist,
            palm_size=palm_size,
            velocity_xy=velocity_xy,
            handedness=handedness,
        )

    def reset(self) -> None:
        self._smoothed_landmarks = None
        self._prev_landmarks = None
        self._prev_ts = None
        self._prev_velocity = None

