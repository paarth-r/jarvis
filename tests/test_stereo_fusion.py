import asyncio
import copy
import numpy as np
import cv2
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.stereo_fusion import StereoFusionModule
from jarvis.events import PoseEvent, WorldPoseEvent

CAMERAS = {
    "left": {
        "device": 0, "width": 1280, "height": 800,
        "intrinsics": {"fx": 910.0, "fy": 910.0, "cx": 640.0, "cy": 400.0},
        "extrinsics": {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [-0.076, 0, 0]},
    },
    "right": {
        "device": 1, "width": 1280, "height": 800,
        "intrinsics": {"fx": 910.0, "fy": 910.0, "cx": 640.0, "cy": 400.0},
        "extrinsics": {"R": [[1,0,0],[0,1,0],[0,0,1]], "t": [0.076, 0, 0]},
    },
}


def _project_world_to_norm(world_pt, cam_cfg):
    """Project a 3D world point to normalized [0-1] image coords."""
    R_c2w = np.array(cam_cfg["extrinsics"]["R"], dtype=np.float64)
    t_c2w = np.array(cam_cfg["extrinsics"]["t"], dtype=np.float64)
    intr = cam_cfg["intrinsics"]
    w, h = cam_cfg["width"], cam_cfg["height"]

    R_wc = R_c2w.T
    t_wc = -R_c2w.T @ t_c2w
    p_cam = R_wc @ np.array(world_pt, dtype=np.float64) + t_wc

    x_px = intr["fx"] * p_cam[0] / p_cam[2] + intr["cx"]
    y_px = intr["fy"] * p_cam[1] / p_cam[2] + intr["cy"]

    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 0] = x_px / w
    lm[:, 1] = y_px / h
    return lm


def _project_distorted(world_pt, cam_cfg):
    """Project a world point to normalized coords WITH lens distortion applied."""
    R_c2w = np.array(cam_cfg["extrinsics"]["R"], dtype=np.float64)
    t_c2w = np.array(cam_cfg["extrinsics"]["t"], dtype=np.float64)
    R_wc = R_c2w.T
    t_wc = -R_c2w.T @ t_c2w
    rvec, _ = cv2.Rodrigues(R_wc)
    intr = cam_cfg["intrinsics"]
    w, h = cam_cfg["width"], cam_cfg["height"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]],
                  [0, 0, 1]], dtype=np.float64)
    d = np.array(cam_cfg["distortion"], dtype=np.float64)
    img, _ = cv2.projectPoints(
        np.array([world_pt], dtype=np.float64).reshape(-1, 1, 3),
        rvec, t_wc.reshape(3, 1), K, d,
    )
    u, v = img.reshape(-1)[:2]
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[:, 0] = u / w
    lm[:, 1] = v / h
    return lm


async def _make_module(bus, cameras=CAMERAS):
    mod = StereoFusionModule("stereo_fusion", {
        "sync_tolerance_ms": 16.0,
        "mono_fallback": True,
        "cameras": cameras,
    })
    await mod.setup(bus)
    return mod


@pytest.mark.asyncio
async def test_full_stereo_triangulates_correctly():
    bus = EventBus()
    received = []

    async def on_world(e: WorldPoseEvent):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    await _make_module(bus)

    world_pt = [0.0, 0.0, 0.5]

    # Publish secondary (right) first so it's buffered when primary (left) arrives
    await bus.publish(PoseEvent(
        camera_id="right",
        landmarks_2d=_project_world_to_norm(world_pt, CAMERAS["right"]),
        timestamp=1.0,
        hand_side="right",
    ))
    await bus.publish(PoseEvent(
        camera_id="left",
        landmarks_2d=_project_world_to_norm(world_pt, CAMERAS["left"]),
        timestamp=1.005,
        hand_side="right",
    ))

    assert len(received) == 1
    assert received[0].confidence == 1.0
    # All 21 landmarks were projected from the same world point; wrist (0) should round-trip
    assert np.allclose(received[0].landmarks_3d[0], world_pt, atol=0.02)


@pytest.mark.asyncio
async def test_undistortion_recovers_distorted_observations():
    cams = copy.deepcopy(CAMERAS)
    cams["left"]["distortion"] = [0.045728, -0.073574, 0.000416, -0.001328, 0.012644]
    cams["right"]["distortion"] = [0.044784, -0.071240, -0.0000363, -0.001105, 0.016704]

    bus = EventBus()
    received = []

    async def on_world(e):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    await _make_module(bus, cameras=cams)

    world_pt = [0.12, 0.08, 0.5]  # off-center so distortion matters
    await bus.publish(PoseEvent(camera_id="right",
                                landmarks_2d=_project_distorted(world_pt, cams["right"]),
                                timestamp=1.0, hand_side="right"))
    await bus.publish(PoseEvent(camera_id="left",
                                landmarks_2d=_project_distorted(world_pt, cams["left"]),
                                timestamp=1.005, hand_side="right"))

    assert len(received) == 1
    assert received[0].confidence == 1.0
    assert np.allclose(received[0].landmarks_3d[0], world_pt, atol=0.01)


@pytest.mark.asyncio
async def test_non_identity_rotation_triangulates():
    # Both cameras tilted 10° about Y (still a valid, consistent stereo geometry).
    theta = np.deg2rad(10.0)
    Ry = [[np.cos(theta), 0, np.sin(theta)],
          [0, 1, 0],
          [-np.sin(theta), 0, np.cos(theta)]]
    cams = copy.deepcopy(CAMERAS)
    cams["left"]["extrinsics"]["R"] = Ry
    cams["right"]["extrinsics"]["R"] = Ry

    bus = EventBus()
    received = []

    async def on_world(e):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    await _make_module(bus, cameras=cams)

    world_pt = [0.05, 0.0, 0.5]
    await bus.publish(PoseEvent(camera_id="right",
                               landmarks_2d=_project_world_to_norm(world_pt, cams["right"]),
                               timestamp=1.0, hand_side="right"))
    await bus.publish(PoseEvent(camera_id="left",
                               landmarks_2d=_project_world_to_norm(world_pt, cams["left"]),
                               timestamp=1.005, hand_side="right"))

    assert len(received) == 1
    assert received[0].confidence == 1.0
    assert np.allclose(received[0].landmarks_3d[0], world_pt, atol=0.02)


@pytest.mark.asyncio
async def test_mono_fallback_when_one_camera_missing():
    bus = EventBus()
    received = []

    async def on_world(e: WorldPoseEvent):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    await _make_module(bus)

    lm = _project_world_to_norm([0.0, 0.0, 0.5], CAMERAS["left"])
    await bus.publish(PoseEvent(
        camera_id="left", landmarks_2d=lm, timestamp=2.0, hand_side="right",
    ))
    # No right camera pose — should mono-fallback
    await asyncio.sleep(0.02)  # wait past sync tolerance

    assert len(received) == 1
    assert received[0].confidence == 0.5


@pytest.mark.asyncio
async def test_single_camera_config_mono_projects():
    # A one-camera rig must mono-project, never attempt triangulation
    bus = EventBus()
    received = []

    async def on_world(e: WorldPoseEvent):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    await _make_module(bus, cameras={"mono": CAMERAS["left"]})

    lm = _project_world_to_norm([0.0, 0.0, 0.5], CAMERAS["left"])
    await bus.publish(PoseEvent(
        camera_id="mono", landmarks_2d=lm, timestamp=1.0, hand_side="right",
    ))

    assert len(received) == 1
    assert received[0].confidence == 0.5


@pytest.mark.asyncio
async def test_out_of_tolerance_poses_not_fused():
    bus = EventBus()
    received = []

    async def on_world(e: WorldPoseEvent):
        received.append(e)

    bus.subscribe(WorldPoseEvent, on_world)
    await _make_module(bus)

    world_pt = [0.0, 0.0, 0.5]

    await bus.publish(PoseEvent(
        camera_id="right",
        landmarks_2d=_project_world_to_norm(world_pt, CAMERAS["right"]),
        timestamp=1.0,
        hand_side="right",
    ))
    # Left arrives 100ms later — outside 16ms tolerance
    await bus.publish(PoseEvent(
        camera_id="left",
        landmarks_2d=_project_world_to_norm(world_pt, CAMERAS["left"]),
        timestamp=1.100,
        hand_side="right",
    ))

    # Fusion should not have happened (left only sees itself → mono fallback)
    assert all(e.confidence < 1.0 for e in received)
