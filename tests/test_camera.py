import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.camera import CameraModule
from jarvis.events import FrameEvent


def _make_module(camera_id="left", device=0):
    return CameraModule(f"camera_{camera_id}", {
        "camera_id": camera_id,
        "device": device,
        "width": 1280,
        "height": 800,
        "fps": 30,
    })


@pytest.mark.asyncio
async def test_camera_publishes_frame_event():
    bus = EventBus()
    received = []

    async def on_frame(e: FrameEvent):
        received.append(e)

    bus.subscribe(FrameEvent, on_frame)

    fake_frame = np.zeros((800, 1280), dtype=np.uint8)

    call_count = [0]

    def fake_read():
        call_count[0] += 1
        if call_count[0] > 2:
            raise asyncio.CancelledError()
        return True, fake_frame

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = fake_read

    mod = _make_module()
    await mod.setup(bus)
    mod._cap = mock_cap

    with pytest.raises((asyncio.CancelledError, Exception)):
        await asyncio.wait_for(mod.run(), timeout=1.0)

    assert len(received) >= 1
    assert received[0].camera_id == "left"
    assert received[0].frame.shape == (800, 1280)


@pytest.mark.asyncio
async def test_camera_raises_if_device_not_open():
    bus = EventBus()
    mod = _make_module()
    await mod.setup(bus)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mod._cap = mock_cap

    with pytest.raises(RuntimeError, match="failed to open"):
        await mod.run()
