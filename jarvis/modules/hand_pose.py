from __future__ import annotations
import asyncio
import os
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from ..events import FrameEvent, PoseEvent
from ..runtime.module_base import Module

# Default model ships with the mouse_control prototype at the repo root.
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "mouse_control", "hand_landmarker.task",
)


class HandPoseModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._camera_id: str = self.config["camera_id"]
        self._model_path: str = self.config.get("model_path", _DEFAULT_MODEL_PATH)
        self._frame_idx: int = 0
        self._landmarker = self._create_landmarker()
        self.bus.subscribe(FrameEvent, self._on_frame)

    def _create_landmarker(self):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self.config.get("min_detection_confidence", 0.7),
            min_hand_presence_confidence=self.config.get("min_detection_confidence", 0.7),
            min_tracking_confidence=self.config.get("min_tracking_confidence", 0.6),
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    async def _on_frame(self, event: FrameEvent) -> None:
        if event.camera_id != self._camera_id:
            return
        loop = asyncio.get_event_loop()
        pose_event = await loop.run_in_executor(None, self._process, event)
        if pose_event is not None:
            await self.bus.publish(pose_event)

    def _process(self, event: FrameEvent) -> Optional[PoseEvent]:
        # OV9281 outputs mono; convert to RGB for MediaPipe
        if event.frame.ndim == 2:
            rgb = cv2.cvtColor(event.frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(event.frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # VIDEO mode requires monotonically increasing timestamps (ms).
        timestamp_ms = int(self._frame_idx * (1000 / 30))
        self._frame_idx += 1

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None

        lm_list = result.hand_landmarks[0]
        hand_side = "right"
        if result.handedness:
            hand_side = result.handedness[0][0].category_name.lower()

        landmarks = np.array(
            [[p.x, p.y, p.z] for p in lm_list], dtype=np.float32
        )
        return PoseEvent(
            camera_id=self._camera_id,
            landmarks_2d=landmarks,
            timestamp=event.timestamp,
            hand_side=hand_side,
        )

    async def run(self) -> None:
        # Subscriptions handle all work; keep coroutine alive
        while True:
            await asyncio.sleep(3600)

    async def teardown(self) -> None:
        self._landmarker.close()
