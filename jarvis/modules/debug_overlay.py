from __future__ import annotations
import asyncio
import time
from typing import Optional

import cv2
import numpy as np

from ..events import FrameEvent, WorldPoseEvent, IntentEvent, SystemEvent
from ..runtime.module_base import Module

# MediaPipe hand connections (landmark index pairs)
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]


class DebugOverlayModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._enabled: bool = self.config.get("enabled", True)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_pose: Optional[WorldPoseEvent] = None
        self._latest_intent: Optional[IntentEvent] = None
        self._fps: float = 0.0
        self._last_frame_ts: float = time.perf_counter()

        if self._enabled:
            self.bus.subscribe(FrameEvent, self._on_frame)
            self.bus.subscribe(WorldPoseEvent, self._on_pose)
            self.bus.subscribe(IntentEvent, self._on_intent)

    async def _on_frame(self, event: FrameEvent) -> None:
        now = time.perf_counter()
        dt = now - self._last_frame_ts
        self._fps = 0.1 * (1.0 / max(dt, 1e-6)) + 0.9 * self._fps
        self._last_frame_ts = now
        self._latest_frame = event.frame.copy()

    async def _on_pose(self, event: WorldPoseEvent) -> None:
        self._latest_pose = event

    async def _on_intent(self, event: IntentEvent) -> None:
        self._latest_intent = event

    async def run(self) -> None:
        if not self._enabled:
            while True:
                await asyncio.sleep(3600)
            return

        loop = asyncio.get_event_loop()
        while True:
            await loop.run_in_executor(None, self._render)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            await asyncio.sleep(0)

        cv2.destroyAllWindows()

    def _render(self) -> None:
        if self._latest_frame is None:
            return

        if self._latest_frame.ndim == 2:
            display = cv2.cvtColor(self._latest_frame, cv2.COLOR_GRAY2BGR)
        else:
            display = self._latest_frame.copy()
        display = cv2.flip(display, 1)
        h, w = display.shape[:2]

        pose = self._latest_pose
        if pose is not None:
            lm = pose.landmarks_3d
            # Back-project to rough pixel coords using first camera (approx)
            lm_px = []
            for i in range(21):
                # Simple depth-normalized projection back to display
                px = int((1.0 - (0.5 + lm[i, 0])) * w)
                py = int((0.5 + lm[i, 1]) * h)
                lm_px.append((px, py))
            for a, b in _CONNECTIONS:
                if 0 <= a < len(lm_px) and 0 <= b < len(lm_px):
                    cv2.line(display, lm_px[a], lm_px[b], (0, 255, 0), 2)
            for pt in lm_px:
                cv2.circle(display, pt, 4, (0, 255, 255), -1)

            conf_str = f"conf:{pose.confidence:.2f}"
            cv2.putText(display, conf_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        intent = self._latest_intent
        if intent is not None:
            cv2.putText(
                display, f"Intent: {intent.type}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        cv2.putText(
            display, f"FPS: {self._fps:.1f}",
            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1,
        )
        cv2.imshow("Jarvis Debug", display)

    async def teardown(self) -> None:
        if self._enabled:
            cv2.destroyAllWindows()
