import os
import tempfile
import pytest
from jarvis.runtime.config import load_config

FULL_CONFIG = """
cameras:
  left:
    device: 0
    width: 1280
    height: 800
    fps: 120
    intrinsics:
      fx: 910.0
      fy: 910.0
      cx: 640.0
      cy: 400.0
    extrinsics:
      R: [[1,0,0],[0,1,0],[0,0,1]]
      t: [-0.076, 0, 0]
  right:
    device: 1
    width: 1280
    height: 800
    fps: 120
    intrinsics:
      fx: 910.0
      fy: 910.0
      cx: 640.0
      cy: 400.0
    extrinsics:
      R: [[1,0,0],[0,1,0],[0,0,1]]
      t: [0.076, 0, 0]

modules:
  stereo_fusion:
    class: jarvis.modules.stereo_fusion.StereoFusionModule
    sync_tolerance_ms: 16.0
    mono_fallback: true
  debug_overlay:
    class: jarvis.modules.debug_overlay.DebugOverlayModule
    enabled: false

subprocesses:
  arm:
    enabled: false
    module: jarvis.processes.arm_controller

hand_pose:
  detection_confidence: 0.5
  tracking_confidence: 0.6
"""


def _write_tmp(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


def test_cameras_loaded():
    path = _write_tmp(FULL_CONFIG)
    try:
        cfg = load_config(path)
        assert set(cfg.cameras.keys()) == {"left", "right"}
        left = cfg.cameras["left"]
        assert left.device == 0
        assert left.width == 1280
        assert left.fps == 120
        assert left.intrinsics["fx"] == 910.0
        assert left.extrinsics["t"] == [-0.076, 0, 0]
    finally:
        os.unlink(path)


def test_modules_loaded():
    path = _write_tmp(FULL_CONFIG)
    try:
        cfg = load_config(path)
        assert "stereo_fusion" in cfg.modules
        sf = cfg.modules["stereo_fusion"]
        assert sf.class_path == "jarvis.modules.stereo_fusion.StereoFusionModule"
        assert sf.enabled is True
        assert sf.settings["sync_tolerance_ms"] == 16.0

        assert cfg.modules["debug_overlay"].enabled is False
    finally:
        os.unlink(path)


def test_subprocesses_loaded():
    path = _write_tmp(FULL_CONFIG)
    try:
        cfg = load_config(path)
        assert "arm" in cfg.subprocesses
        assert cfg.subprocesses["arm"].enabled is False
        assert cfg.subprocesses["arm"].module == "jarvis.processes.arm_controller"
    finally:
        os.unlink(path)


def test_hand_pose_loaded():
    path = _write_tmp(FULL_CONFIG)
    try:
        cfg = load_config(path)
        assert cfg.hand_pose["detection_confidence"] == 0.5
        assert cfg.hand_pose["tracking_confidence"] == 0.6
    finally:
        os.unlink(path)


def test_hand_pose_defaults_to_empty_when_absent():
    path = _write_tmp("cameras: {}\nmodules: {}\nsubprocesses: {}\n")
    try:
        cfg = load_config(path)
        assert cfg.hand_pose == {}
    finally:
        os.unlink(path)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/file.yaml")
