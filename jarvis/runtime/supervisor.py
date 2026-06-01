from __future__ import annotations
import asyncio
import importlib
import logging
from typing import List

from .event_bus import EventBus
from .module_base import Module
from .config import JarvisConfig
from .subprocess_bridge import SubprocessBridge
from ..events import SystemEvent

logger = logging.getLogger(__name__)


class Supervisor:
    def __init__(self, config: JarvisConfig) -> None:
        self._config = config
        self._bus = EventBus()
        self._modules: List[Module] = []
        self._bridges: List[SubprocessBridge] = []

    # ── internal helper used by tests ──────────────────────────────────────
    def _inject_module(self, module: Module) -> None:
        self._modules.append(module)

    # ── public ─────────────────────────────────────────────────────────────
    async def start(self) -> None:
        self._build_modules()

        for mod in self._modules:
            await mod.setup(self._bus)

        for name, sub_cfg in self._config.subprocesses.items():
            if sub_cfg.enabled:
                bridge = SubprocessBridge(name, sub_cfg.module, self._bus)
                self._bridges.append(bridge)

        tasks = [asyncio.create_task(mod.run()) for mod in self._modules]
        tasks += [asyncio.create_task(b.run()) for b in self._bridges]

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except (asyncio.CancelledError, Exception) as exc:
            if not isinstance(exc, asyncio.CancelledError):
                logger.exception("Module raised an exception")
                await self._bus.publish(SystemEvent(
                    type="module_error",
                    module_id="supervisor",
                    payload={"error": str(exc)},
                ))
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for mod in reversed(self._modules):
                try:
                    await mod.teardown()
                except Exception:
                    logger.exception("teardown error in %s", mod.module_id)

    @property
    def bus(self) -> EventBus:
        return self._bus

    # ── private ────────────────────────────────────────────────────────────
    def _build_modules(self) -> None:
        from ..modules.camera import CameraModule
        from ..modules.hand_pose import HandPoseModule

        # One CameraModule + HandPoseModule per declared camera
        cameras_dict = {
            cam_id: {
                "camera_id": cam_id,
                "device": cam.device,
                "width": cam.width,
                "height": cam.height,
                "fps": cam.fps,
                "intrinsics": cam.intrinsics,
                "extrinsics": cam.extrinsics,
            }
            for cam_id, cam in self._config.cameras.items()
        }

        for cam_id, cam in self._config.cameras.items():
            self._modules.append(CameraModule(f"camera_{cam_id}", cameras_dict[cam_id]))
            self._modules.append(HandPoseModule(f"hand_pose_{cam_id}", {"camera_id": cam_id}))

        # Declared modules — inject cameras into every module's config
        for mod_id, mod_cfg in self._config.modules.items():
            if not mod_cfg.enabled:
                continue
            cls = self._load_class(mod_cfg.class_path)
            full_cfg = {**mod_cfg.settings, "cameras": cameras_dict}
            self._modules.append(cls(mod_id, full_cfg))

    @staticmethod
    def _load_class(class_path: str):
        module_path, class_name = class_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
