import numpy as np
import pytest
from jarvis.events import (
    FrameEvent, PoseEvent, WorldPoseEvent,
    IntentEvent, PointEvent, ActionEvent, SystemEvent,
)

def test_frame_event_fields():
    frame = np.zeros((800, 1280), dtype=np.uint8)
    e = FrameEvent(camera_id="left", frame=frame, timestamp=1.0)
    assert e.camera_id == "left"
    assert e.frame.shape == (800, 1280)
    assert e.timestamp == 1.0

def test_world_pose_event_fields():
    lm = np.zeros((21, 3), dtype=np.float32)
    e = WorldPoseEvent(landmarks_3d=lm, timestamp=2.0, confidence=1.0, hand_side="right")
    assert e.confidence == 1.0
    assert e.landmarks_3d.shape == (21, 3)

def test_intent_event_fields():
    e = IntentEvent(type="click", confidence=0.95, payload={"x": 0.5}, timestamp=3.0)
    assert e.type == "click"
    assert e.payload["x"] == 0.5
