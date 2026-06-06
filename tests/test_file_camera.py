import asyncio
import numpy as np
import cv2
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.file_camera import FileCameraModule
from jarvis.events import FrameEvent


def _write_images(tmp_path, n=3):
    for i in range(n):
        img = np.full((48, 64, 3), i * 10, dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"), img)
    return str(tmp_path)


@pytest.mark.asyncio
async def test_replays_image_directory_once(tmp_path):
    path = _write_images(tmp_path, 3)
    bus = EventBus()
    received = []

    async def on_frame(e: FrameEvent):
        received.append(e)

    bus.subscribe(FrameEvent, on_frame)

    mod = FileCameraModule("camera_left", {
        "camera_id": "left", "path": path, "fps": 120, "loop": False,
    })
    await mod.setup(bus)
    await asyncio.wait_for(mod.run(), timeout=2.0)

    assert len(received) == 3
    assert all(e.camera_id == "left" for e in received)
    assert received[0].frame.shape[:2] == (48, 64)


@pytest.mark.asyncio
async def test_empty_directory_raises(tmp_path):
    bus = EventBus()
    mod = FileCameraModule("camera_left", {
        "camera_id": "left", "path": str(tmp_path), "fps": 120, "loop": False,
    })
    await mod.setup(bus)
    with pytest.raises(RuntimeError, match="No images"):
        await mod.run()
