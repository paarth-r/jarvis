import numpy as np
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.visualizer import VisualizerModule
from jarvis.events import FrameEvent, PoseEvent, WorldPoseEvent


def _make(show_2d=False, show_3d=False):
    return VisualizerModule("visualizer", {
        "enabled": True,
        "show_2d": show_2d,
        "show_3d": show_3d,
        "fps": 30,
        "cameras": {"left": {"width": 1280, "height": 800},
                    "right": {"width": 1280, "height": 800}},
    })


@pytest.mark.asyncio
async def test_headless_setup_does_not_spawn_or_raise():
    bus = EventBus()
    mod = _make(show_2d=False, show_3d=False)
    await mod.setup(bus)
    assert mod._viewer is None  # show_3d False → no subprocess


@pytest.mark.asyncio
async def test_caches_latest_events():
    bus = EventBus()
    mod = _make(show_2d=False, show_3d=False)
    await mod.setup(bus)

    frame = np.zeros((800, 1280), dtype=np.uint8)
    await bus.publish(FrameEvent(camera_id="left", frame=frame, timestamp=1.0))
    lm2d = np.full((21, 3), 0.5, dtype=np.float32)
    await bus.publish(PoseEvent(camera_id="left", landmarks_2d=lm2d, timestamp=1.0, hand_side="right"))
    lm3d = np.zeros((21, 3), dtype=np.float32)
    await bus.publish(WorldPoseEvent(landmarks_3d=lm3d, timestamp=1.0, confidence=1.0, hand_side="right"))

    assert mod._frames["left"] is not None
    assert mod._poses_2d["left"].shape == (21, 3)
    assert mod._world is not None
    assert mod._confidence == 1.0


@pytest.mark.asyncio
async def test_disabled_module_skips_subscriptions():
    bus = EventBus()
    mod = VisualizerModule("visualizer", {"enabled": False})
    await mod.setup(bus)
    # No crash; publishing does nothing harmful
    await bus.publish(WorldPoseEvent(landmarks_3d=np.zeros((21, 3), dtype=np.float32),
                                     timestamp=1.0, confidence=1.0, hand_side="right"))
    assert mod._world is None
