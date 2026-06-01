from __future__ import annotations
import asyncio
import time
from typing import Optional

import cv2
import numpy as np

from ..events import FrameEvent
from ..runtime.module_base import Module


class CameraModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._camera_id: str = self.config["camera_id"]
        self._device: int = self.config["device"]
        self._width: int = self.config.get("width", 1280)
        self._height: int = self.config.get("height", 800)
        self._fps: int = self.config.get("fps", 30)
        self._cap: Optional[cv2.VideoCapture] = None

    async def run(self) -> None:
        loop = asyncio.get_event_loop()
        if self._cap is None:
            self._cap = await loop.run_in_executor(None, self._open_capture)
        if not self._cap.isOpened():
            raise RuntimeError(f"Camera device {self._device} failed to open")

        try:
            while True:
                frame = await loop.run_in_executor(None, self._read_frame)
                if frame is None:
                    await asyncio.sleep(0.001)
                    continue
                await self.bus.publish(FrameEvent(
                    camera_id=self._camera_id,
                    frame=frame,
                    timestamp=time.perf_counter(),
                ))
        finally:
            if self._cap:
                self._cap.release()
                self._cap = None

    async def teardown(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        return cap

    def _read_frame(self) -> Optional[np.ndarray]:
        if not self._cap:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None
