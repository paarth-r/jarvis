from __future__ import annotations
import asyncio
import glob
import os
import time
from typing import List, Optional

import cv2

from ..events import FrameEvent
from ..runtime.module_base import Module


class FileCameraModule(Module):
    """Replay frames from a video file or a directory of images as FrameEvents.

    Drop-in alternative to CameraModule that needs no live hardware — point it at
    a recorded clip or an image folder and the rest of the pipeline runs as usual.
    """

    _IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._camera_id: str = self.config["camera_id"]
        self._path: str = self.config["path"]
        self._fps: int = self.config.get("fps", 30)
        self._loop: bool = self.config.get("loop", True)
        self._images: Optional[List[str]] = None
        if os.path.isdir(self._path):
            files: List[str] = []
            for ext in self._IMAGE_EXTS:
                files.extend(glob.glob(os.path.join(self._path, ext)))
            self._images = sorted(files)

    async def run(self) -> None:
        period = 1.0 / max(self._fps, 1)
        if self._images is not None:
            await self._run_images(period)
        else:
            await self._run_video(period)

    async def _run_images(self, period: float) -> None:
        if not self._images:
            raise RuntimeError(f"No images found in {self._path}")
        loop = asyncio.get_event_loop()
        while True:
            for p in self._images:
                frame = await loop.run_in_executor(None, cv2.imread, p, cv2.IMREAD_UNCHANGED)
                if frame is None:
                    continue
                await self.bus.publish(FrameEvent(
                    camera_id=self._camera_id, frame=frame, timestamp=time.perf_counter(),
                ))
                await asyncio.sleep(period)
            if not self._loop:
                return

    async def _run_video(self, period: float) -> None:
        loop = asyncio.get_event_loop()
        while True:
            cap = await loop.run_in_executor(None, cv2.VideoCapture, self._path)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open video {self._path}")
            try:
                while True:
                    ok, frame = await loop.run_in_executor(None, cap.read)
                    if not ok:
                        break
                    await self.bus.publish(FrameEvent(
                        camera_id=self._camera_id, frame=frame, timestamp=time.perf_counter(),
                    ))
                    await asyncio.sleep(period)
            finally:
                cap.release()
            if not self._loop:
                return
