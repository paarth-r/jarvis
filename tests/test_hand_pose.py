import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.hand_pose import HandPoseModule
from jarvis.events import FrameEvent, PoseEvent


def _make_pose_module(camera_id="left"):
    mod = HandPoseModule(f"hand_pose_{camera_id}", {"camera_id": camera_id})
    # Avoid loading the real MediaPipe model in unit tests (mirrors how the
    # camera test stubs cv2.VideoCapture). A real landmarker leaked to GC at
    # interpreter shutdown deadlocks in mediapipe's close().
    mod._create_landmarker = lambda: MagicMock()
    return mod


@pytest.mark.asyncio
async def test_wrong_camera_id_ignored():
    bus = EventBus()
    received = []

    async def on_pose(e: PoseEvent):
        received.append(e)

    bus.subscribe(PoseEvent, on_pose)

    mod = _make_pose_module(camera_id="left")
    await mod.setup(bus)

    # Publish a frame from "right" — should be ignored
    frame = FrameEvent(camera_id="right", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(frame)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_no_hand_detected_publishes_nothing():
    bus = EventBus()
    received = []

    async def on_pose(e: PoseEvent):
        received.append(e)

    bus.subscribe(PoseEvent, on_pose)

    mod = _make_pose_module(camera_id="left")

    # Patch _process to return None (no hand found)
    mod._process = lambda event: None
    await mod.setup(bus)

    frame = FrameEvent(camera_id="left", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(frame)

    assert len(received) == 0


@pytest.mark.asyncio
async def test_hand_detected_publishes_pose_event():
    bus = EventBus()
    received = []

    async def on_pose(e: PoseEvent):
        received.append(e)

    bus.subscribe(PoseEvent, on_pose)

    mod = _make_pose_module(camera_id="left")

    fake_landmarks = np.zeros((21, 3), dtype=np.float32)
    fake_pose = PoseEvent(
        camera_id="left",
        landmarks_2d=fake_landmarks,
        timestamp=1.0,
        hand_side="right",
    )
    mod._process = lambda event: fake_pose
    await mod.setup(bus)

    frame = FrameEvent(camera_id="left", frame=np.zeros((800, 1280), dtype=np.uint8), timestamp=1.0)
    await bus.publish(frame)

    assert len(received) == 1
    assert received[0].camera_id == "left"
