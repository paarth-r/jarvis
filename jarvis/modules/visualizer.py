from __future__ import annotations
import asyncio
import logging
from typing import Dict, Optional

import numpy as np

from ..events import FrameEvent, PoseEvent, WorldPoseEvent
from ..runtime.module_base import Module
from .hand_landmarks import HAND_CONNECTIONS, hand_points, draw_landmarks

logger = logging.getLogger(__name__)


class VisualizerModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._enabled: bool = self.config.get("enabled", True)
        self._show_2d: bool = self.config.get("show_2d", True)
        self._show_3d: bool = self.config.get("show_3d", True)
        self._fps: int = self.config.get("fps", 30)
        self._cameras: Dict[str, dict] = self.config.get("cameras", {})

        self._frames: Dict[str, Optional[np.ndarray]] = {}
        self._poses_2d: Dict[str, Optional[np.ndarray]] = {}
        self._world: Optional[np.ndarray] = None
        self._confidence: float = 0.0
        self._viewer = None

        if not self._enabled:
            return

        if self._show_3d:
            from .viewer_3d import Viewer3D
            self._viewer = Viewer3D()
            self._viewer.start()

        self.bus.subscribe(FrameEvent, self._on_frame)
        self.bus.subscribe(PoseEvent, self._on_pose)
        self.bus.subscribe(WorldPoseEvent, self._on_world)

    async def _on_frame(self, event: FrameEvent) -> None:
        self._frames[event.camera_id] = event.frame

    async def _on_pose(self, event: PoseEvent) -> None:
        self._poses_2d[event.camera_id] = event.landmarks_2d

    async def _on_world(self, event: WorldPoseEvent) -> None:
        self._world = event.landmarks_3d
        self._confidence = event.confidence

    async def run(self) -> None:
        if not self._enabled or not (self._show_2d or self._show_3d):
            while True:
                await asyncio.sleep(3600)

        import cv2
        period = 1.0 / max(self._fps, 1)
        try:
            while True:
                if self._show_2d:
                    for cam_id, frame in list(self._frames.items()):
                        if frame is None:
                            continue
                        lm2d = self._poses_2d.get(cam_id)
                        if lm2d is not None:
                            img = draw_landmarks(frame, lm2d, HAND_CONNECTIONS)
                        elif frame.ndim == 2:
                            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        else:
                            img = frame
                        cv2.putText(img, f"{cam_id} conf:{self._confidence:.2f}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2)
                        cv2.imshow(f"jarvis {cam_id}", img)
                if self._show_3d and self._viewer is not None:
                    self._viewer.update(hand_points(self._world))
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
                await asyncio.sleep(period)
        finally:
            cv2.destroyAllWindows()

    async def teardown(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
