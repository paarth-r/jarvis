from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, Optional

from ..events import ActionEvent, GestureEvent, StudioModeEvent
from ..gestures.actions import ActionRunner, normalize
from ..gestures.library import GestureLibrary
from ..runtime.module_base import Module

logger = logging.getLogger(__name__)


class ActionDispatcher(Module):
    """Runs the action bound to a gesture when the gesture starts (the
    arbiter in gesture_hci decides when that is). Bindings come from
    ~/.jarvis/gestures.json (edited in the studio, picked up live) over the
    config's defaults."""

    async def setup(self, bus) -> None:
        await super().setup(bus)
        # Minimum gap between two firings of the same gesture's action.
        self._cooldown: float = self.config.get("cooldown_s", 1.0)
        # Config defaults; on_point_action is the legacy spelling of
        # bindings.point ("" / None disables it).
        self._defaults: Dict[str, Any] = {
            "point": self.config.get("on_point_action", "raycast"),
            **(self.config.get("bindings") or {}),
        }
        self.library = GestureLibrary(self.config.get("library_path"))
        self.runner = ActionRunner()
        self._last: Dict[str, float] = {}
        self._repeats: Dict[str, asyncio.Task] = {}
        self.paused = False      # the studio pauses actions while recording
        self.bus.subscribe(ActionEvent, self._on_action)
        self.bus.subscribe(GestureEvent, self._on_gesture)
        self.bus.subscribe(StudioModeEvent, self._on_studio_mode)

    def binding(self, gesture: str) -> Optional[Dict[str, Any]]:
        self.library.reload_if_changed()
        if gesture in self.library.bindings:
            return normalize(self.library.bindings[gesture])
        return normalize(self._defaults.get(gesture))

    def _fire(self, key: str, action: Any, cooldown: bool = True) -> bool:
        """Run an action unless `key` fired within the cooldown. True if it ran."""
        now = time.perf_counter()
        if cooldown and now - self._last.get(key, -float("inf")) < self._cooldown:
            return False
        self._last[key] = now
        if ActionRunner.is_slow(action):
            # Macro waits must not stall the gesture loop.
            asyncio.get_running_loop().run_in_executor(None, self.runner.run, action)
            return True
        return self.runner.run(action)

    async def _on_action(self, event: ActionEvent) -> None:
        self._fire(f"action:{event.action_type}", event.action_type)

    async def _on_gesture(self, event: GestureEvent) -> None:
        if event.phase == "end":
            task = self._repeats.pop(event.name, None)
            if task:
                task.cancel()
            return
        if self.paused:
            return
        action = self.binding(event.name)
        if not action or not self._fire(event.name, action):
            return
        repeat = action.get("repeat_s")
        if repeat:
            self._repeats[event.name] = asyncio.create_task(self._repeat(event.name, action, repeat))

    async def _repeat(self, name: str, action: dict, every: float) -> None:
        while True:
            await asyncio.sleep(every)
            if self.paused:
                return
            self._fire(name, action, cooldown=False)

    async def _on_studio_mode(self, event: StudioModeEvent) -> None:
        self.paused = event.recording
        if self.paused:
            for task in self._repeats.values():
                task.cancel()
            self._repeats.clear()

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)
