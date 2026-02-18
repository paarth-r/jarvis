"""
Frame capture, FPS control, timestamps.
Resolution 1280x800, max available FPS, frame delta-time.
"""
import cv2
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Frame:
    """Single captured frame with metadata."""
    image: "cv2.Mat"
    timestamp: float
    dt: float  # seconds since previous frame


class Camera:
    def __init__(self, width: int = 1280, height: int = 800, device_id: int = 0):
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None
        self._device_id = device_id
        self._prev_ts: Optional[float] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, 120)  # request max FPS
        self._prev_ts = None
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ret, image = self._cap.read()
        if not ret:
            return None
        ts = time.perf_counter()
        dt = (ts - self._prev_ts) if self._prev_ts is not None else 0.0
        self._prev_ts = ts
        return Frame(image=image, timestamp=ts, dt=dt)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._prev_ts = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.release()
