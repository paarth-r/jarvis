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
async def test_detection_confidence_defaults_to_0_5():
    bus = EventBus()
    mod = _make_pose_module(camera_id="left")
    await mod.setup(bus)
    assert mod._detection_confidence == 0.5
    assert mod._tracking_confidence == 0.6


@pytest.mark.asyncio
async def test_detection_confidence_from_config():
    bus = EventBus()
    mod = HandPoseModule("hand_pose_left", {"camera_id": "left", "detection_confidence": 0.3})
    mod._create_landmarker = lambda: MagicMock()
    await mod.setup(bus)
    assert mod._detection_confidence == 0.3


@pytest.mark.asyncio
async def test_process_accepts_both_mono_and_color_frames():
    bus = EventBus()
    mod = _make_pose_module(camera_id="left")

    class _NoHand:
        hand_landmarks = []
        handedness = []

    fake = MagicMock()
    fake.detect_for_video.return_value = _NoHand()
    mod._create_landmarker = lambda: fake
    await mod.setup(bus)

    mono = np.zeros((800, 1280), dtype=np.uint8)          # HxW mono
    color = np.zeros((800, 1280, 3), dtype=np.uint8)      # HxWx3 BGR color
    assert mod._process(FrameEvent(camera_id="left", frame=mono, timestamp=1.0)) is None
    assert mod._process(FrameEvent(camera_id="left", frame=color, timestamp=2.0)) is None
    assert fake.detect_for_video.call_count == 2


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
