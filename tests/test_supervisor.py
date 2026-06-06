import asyncio
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.runtime.module_base import Module
from jarvis.runtime.supervisor import Supervisor
from jarvis.runtime.config import JarvisConfig, CameraConfig, ModuleConfig, SubprocessConfig


def _empty_config() -> JarvisConfig:
    return JarvisConfig(cameras={}, modules={}, subprocesses={})


class _LifecycleModule(Module):
    def __init__(self, module_id, config):
        super().__init__(module_id, config)
        self.setup_called = False
        self.run_called = False
        self.teardown_called = False

    async def setup(self, bus):
        await super().setup(bus)
        self.setup_called = True

    async def run(self):
        self.run_called = True
        # Immediately exit so gather() completes
        return

    async def teardown(self):
        self.teardown_called = True


@pytest.mark.asyncio
async def test_supervisor_calls_lifecycle():
    cfg = _empty_config()
    sup = Supervisor(cfg)

    mod = _LifecycleModule("test_mod", {})
    sup._inject_module(mod)

    await sup.start()

    assert mod.setup_called
    assert mod.run_called
    assert mod.teardown_called


def test_build_modules_injects_hand_pose_config():
    from jarvis.modules.hand_pose import HandPoseModule
    cam = CameraConfig(device=0, width=1280, height=800, fps=30, intrinsics={}, extrinsics={})
    cfg = JarvisConfig(
        cameras={"left": cam}, modules={}, subprocesses={},
        hand_pose={"detection_confidence": 0.3, "tracking_confidence": 0.4},
    )
    sup = Supervisor(cfg)
    sup._build_modules()
    hp = next(m for m in sup._modules if isinstance(m, HandPoseModule))
    assert hp.config["camera_id"] == "left"
    assert hp.config["detection_confidence"] == 0.3
    assert hp.config["tracking_confidence"] == 0.4


@pytest.mark.asyncio
async def test_supervisor_bus_is_shared():
    cfg = _empty_config()
    sup = Supervisor(cfg)

    buses = []

    class _BusCaptureModule(Module):
        async def setup(self, bus):
            await super().setup(bus)
            buses.append(bus)
        async def run(self):
            return

    sup._inject_module(_BusCaptureModule("a", {}))
    sup._inject_module(_BusCaptureModule("b", {}))
    await sup.start()

    assert len(buses) == 2
    assert buses[0] is buses[1]
