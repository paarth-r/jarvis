from __future__ import annotations
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from ..events import FrameEvent, GazeEvent
from ..runtime.module_base import Module
from .gaze_model import DEFAULT_CALIBRATION_PATH, GazeMapper, gaze_features
from .hand_pose import _get_shared_executor
from .one_euro import OneEuroFilter

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "mouse_control", "face_landmarker.task",
)


class FaceTracker:
    """MediaPipe Face Landmarker (iris-refined 478-point mesh + head transform)
    in VIDEO mode. Not thread-safe; call from one thread."""

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH) -> None:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def features(self, bgr: np.ndarray, timestamp_s: float) -> Optional[np.ndarray]:
        """Gaze feature vector for the face in a BGR frame, or None."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # VIDEO mode requires strictly increasing timestamps (ms).
        ts_ms = max(int(timestamp_s * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.face_landmarks:
            return None
        lm = np.array([[p.x, p.y, p.z] for p in result.face_landmarks[0]])
        matrix = None
        if result.facial_transformation_matrixes:
            matrix = np.asarray(result.facial_transformation_matrixes[0])
        return gaze_features(lm, matrix)

    def close(self) -> None:
        self._landmarker.close()


class GazeModule(Module):
    """FrameEvent -> GazeEvent: where on the calibrated display the user is
    looking, normalized 0-1 and 1-euro smoothed. Needs a calibration file
    from `jarvis-gaze-calibrate`."""

    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._camera_id: str = self.config.get("camera_id", "main")
        path = Path(os.path.expanduser(self.config.get("calibration_path", str(DEFAULT_CALIBRATION_PATH))))
        if not path.exists():
            raise FileNotFoundError(
                f"No gaze calibration at {path} - run `jarvis-gaze-calibrate` first"
            )
        self._mapper = GazeMapper.load(path)
        self._filter = OneEuroFilter(
            min_cutoff=self.config.get("min_cutoff", 0.4),
            beta=self.config.get("beta", 2.0),
        )
        self._busy = False
        # Same worker thread as HandPoseModule: MediaPipe GPU contexts must
        # never run concurrently on macOS Metal.
        self._executor = _get_shared_executor()
        loop = asyncio.get_event_loop()
        model_path = self.config.get("model_path", _DEFAULT_MODEL_PATH)
        self._tracker = await loop.run_in_executor(self._executor, FaceTracker, model_path)
        self.bus.subscribe(FrameEvent, self._on_frame)

    async def _on_frame(self, event: FrameEvent) -> None:
        if event.camera_id != self._camera_id or self._busy:
            return
        self._busy = True
        loop = asyncio.get_event_loop()
        try:
            feats = await loop.run_in_executor(
                self._executor, self._tracker.features, event.frame, event.timestamp
            )
        finally:
            self._busy = False
        if feats is None:
            return
        point = np.clip(self._mapper.predict(feats), 0.0, 1.0)
        x, y = self._filter(point, event.timestamp)
        await self.bus.publish(GazeEvent(
            norm_x=float(x), norm_y=float(y), timestamp=event.timestamp, confidence=1.0,
        ))

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)

    async def teardown(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._tracker.close)


def main_display_size() -> Tuple[int, int]:
    """Main display size in points (the space Quartz cursor events use)."""
    from Quartz.CoreGraphics import CGDisplayBounds, CGMainDisplayID
    b = CGDisplayBounds(CGMainDisplayID())
    return int(b.size.width), int(b.size.height)
