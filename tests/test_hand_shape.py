"""MediaPipe's hand_world_landmarks (metric, hand-centred, with depth) ride
from HandPoseModule through StereoFusion as WorldPoseEvent.hand_shape, so shape
features and tilt work from a single camera (mono projection flattens z)."""
from unittest.mock import MagicMock

import numpy as np
import pytest

from jarvis.events import FrameEvent, PoseEvent, WorldPoseEvent
from jarvis.runtime.event_bus import EventBus
from tests.test_hand_pose import _make_pose_module
from tests.test_stereo_fusion import CAMERAS, _make_module, _project_world_to_norm


async def _pose_module_returning(hand):
    mod = _make_pose_module(camera_id="left")
    fake = MagicMock()
    fake.detect_for_video.return_value = hand
    mod._create_landmarker = lambda: fake
    await mod.setup(EventBus())
    return mod


def _frame():
    return FrameEvent(camera_id="left", frame=np.zeros((8, 8), np.uint8), timestamp=1.0)


def _pts(scale):
    return [MagicMock(x=i * scale, y=-i * scale, z=0.5 * i * scale) for i in range(21)]


@pytest.mark.asyncio
async def test_process_carries_world_landmarks():
    class _Hand:
        hand_landmarks = [_pts(0.01)]
        hand_world_landmarks = [_pts(0.002)]
        handedness = [[MagicMock(category_name="Right")]]

    mod = await _pose_module_returning(_Hand())
    pose = mod._process(_frame())
    assert pose.world_landmarks.shape == (21, 3)
    np.testing.assert_allclose(pose.world_landmarks[20], [0.04, -0.04, 0.02], atol=1e-6)


@pytest.mark.asyncio
async def test_process_world_landmarks_none_when_missing():
    class _Hand:
        hand_landmarks = [_pts(0.01)]
        hand_world_landmarks = []
        handedness = []

    mod = await _pose_module_returning(_Hand())
    assert mod._process(_frame()).world_landmarks is None


async def _collect(bus):
    received = []

    async def on_world(e: WorldPoseEvent):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    return received


@pytest.mark.asyncio
async def test_mono_passes_hand_shape_through():
    bus = EventBus()
    received = await _collect(bus)
    await _make_module(bus, cameras={"mono": CAMERAS["left"]})

    shape = np.random.default_rng(0).normal(size=(21, 3)).astype(np.float32)
    lm = _project_world_to_norm([0.0, 0.0, 0.5], CAMERAS["left"])
    await bus.publish(PoseEvent(camera_id="mono", landmarks_2d=lm, timestamp=1.0,
                                hand_side="right", world_landmarks=shape))
    np.testing.assert_array_equal(received[0].hand_shape, shape)


@pytest.mark.asyncio
async def test_stereo_passes_primary_hand_shape_through():
    bus = EventBus()
    received = await _collect(bus)
    await _make_module(bus)

    shape = np.ones((21, 3), dtype=np.float32)
    pt = [0.05, -0.02, 0.6]
    await bus.publish(PoseEvent(camera_id="right", timestamp=1.0, hand_side="right",
                                landmarks_2d=_project_world_to_norm(pt, CAMERAS["right"])))
    await bus.publish(PoseEvent(camera_id="left", timestamp=1.0, hand_side="right",
                                landmarks_2d=_project_world_to_norm(pt, CAMERAS["left"]),
                                world_landmarks=shape))
    assert received[0].confidence == 1.0
    np.testing.assert_array_equal(received[0].hand_shape, shape)


def test_hand_shape_defaults_to_none():
    # Existing producers (replay, tests) that never set it keep working.
    e = WorldPoseEvent(landmarks_3d=np.zeros((21, 3)), timestamp=0.0, confidence=1.0, hand_side="right")
    assert e.hand_shape is None
