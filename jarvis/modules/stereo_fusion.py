from __future__ import annotations
import asyncio
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..events import PoseEvent, WorldPoseEvent
from ..runtime.module_base import Module


class StereoFusionModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._tolerance: float = self.config.get("sync_tolerance_ms", 16.0) / 1000.0
        self._mono_fallback: bool = self.config.get("mono_fallback", True)
        self._mono_depth: float = self.config.get("mono_depth_m", 0.5)
        self._camera_configs: Dict[str, dict] = self.config.get("cameras", {})

        camera_ids = list(self._camera_configs.keys())
        self._primary: str = camera_ids[0] if camera_ids else ""
        self._buffers: Dict[str, deque] = {cid: deque(maxlen=20) for cid in camera_ids}
        self._last_fused_ts: float = -float("inf")

        self.bus.subscribe(PoseEvent, self._on_pose)

    async def _on_pose(self, event: PoseEvent) -> None:
        if event.camera_id not in self._buffers:
            return
        self._buffers[event.camera_id].append(event)

        # Only the primary camera triggers fusion to avoid duplicates
        if event.camera_id != self._primary:
            return

        world_event = self._try_fuse(event.timestamp)
        if world_event is not None:
            self._last_fused_ts = world_event.timestamp
            await self.bus.publish(world_event)

    def _try_fuse(self, ref_ts: float) -> Optional[WorldPoseEvent]:
        camera_ids = list(self._camera_configs.keys())
        matched: Dict[str, PoseEvent] = {}
        for cid in camera_ids:
            best = self._closest(cid, ref_ts)
            if best is not None:
                matched[cid] = best

        if len(matched) == len(camera_ids):
            return self._triangulate(matched)

        if self._mono_fallback and matched:
            cid = next(iter(matched))
            return self._mono_project(matched[cid])

        return None

    def _closest(self, cam_id: str, ref_ts: float) -> Optional[PoseEvent]:
        best: Optional[PoseEvent] = None
        best_dt = float("inf")
        for event in self._buffers.get(cam_id, []):
            dt = abs(event.timestamp - ref_ts)
            if dt < best_dt and dt <= self._tolerance:
                best_dt = dt
                best = event
        return best

    def _build_proj(self, cam_cfg: dict) -> np.ndarray:
        intr = cam_cfg["intrinsics"]
        K = np.array([
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # Config stores camera-to-world; convert to world-to-camera for cv2
        R_c2w = np.array(cam_cfg["extrinsics"]["R"], dtype=np.float64)
        t_c2w = np.array(cam_cfg["extrinsics"]["t"], dtype=np.float64)
        R_wc = R_c2w.T
        t_wc = (-R_c2w.T @ t_c2w).reshape(3, 1)
        return K @ np.hstack([R_wc, t_wc])

    def _triangulate(self, matched: Dict[str, PoseEvent]) -> WorldPoseEvent:
        ids = list(matched.keys())
        cam_a, cam_b = ids[0], ids[1]
        cfg_a, cfg_b = self._camera_configs[cam_a], self._camera_configs[cam_b]

        P1 = self._build_proj(cfg_a)
        P2 = self._build_proj(cfg_b)

        def to_px(lm2d: np.ndarray, cfg: dict) -> np.ndarray:
            return (lm2d[:, :2] * np.array([cfg["width"], cfg["height"]])).T.astype(np.float64)

        pts_a = to_px(matched[cam_a].landmarks_2d, cfg_a)
        pts_b = to_px(matched[cam_b].landmarks_2d, cfg_b)

        pts4d = cv2.triangulatePoints(P1, P2, pts_a, pts_b)
        pts3d = (pts4d[:3] / pts4d[3]).T.astype(np.float32)

        avg_ts = (matched[cam_a].timestamp + matched[cam_b].timestamp) / 2.0
        return WorldPoseEvent(
            landmarks_3d=pts3d,
            timestamp=avg_ts,
            confidence=1.0,
            hand_side=matched[cam_a].hand_side,
        )

    def _mono_project(self, event: PoseEvent) -> WorldPoseEvent:
        cfg = self._camera_configs[event.camera_id]
        intr = cfg["intrinsics"]
        w, h = cfg["width"], cfg["height"]
        d = self._mono_depth

        lm_px = event.landmarks_2d[:, :2] * np.array([w, h])
        lm3d = np.zeros((21, 3), dtype=np.float32)
        lm3d[:, 0] = (lm_px[:, 0] - intr["cx"]) * d / intr["fx"]
        lm3d[:, 1] = (lm_px[:, 1] - intr["cy"]) * d / intr["fy"]
        lm3d[:, 2] = d

        return WorldPoseEvent(
            landmarks_3d=lm3d,
            timestamp=event.timestamp,
            confidence=0.5,
            hand_side=event.hand_side,
        )

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)
