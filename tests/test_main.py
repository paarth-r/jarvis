import asyncio
import pytest
from unittest.mock import patch, MagicMock
from jarvis.main import build_supervisor


def test_build_supervisor_returns_supervisor(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("""
cameras:
  left:
    device: 0
    width: 1280
    height: 800
    fps: 120
    intrinsics: {fx: 910.0, fy: 910.0, cx: 640.0, cy: 400.0}
    extrinsics:
      R: [[1,0,0],[0,1,0],[0,0,1]]
      t: [-0.076, 0, 0]

modules:
  debug_overlay:
    class: jarvis.modules.debug_overlay.DebugOverlayModule
    enabled: false

subprocesses:
  arm:
    enabled: false
    module: jarvis.processes.arm_controller
""")
    from jarvis.runtime.supervisor import Supervisor
    sup = build_supervisor(str(cfg_file))
    assert isinstance(sup, Supervisor)
