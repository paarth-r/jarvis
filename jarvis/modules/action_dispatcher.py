from __future__ import annotations
import asyncio
import logging
import time
from typing import Callable, Dict

import pyautogui
from pynput.keyboard import Controller, Key

from ..events import ActionEvent, GestureEvent
from ..runtime.module_base import Module

logger = logging.getLogger(__name__)
_keyboard = Controller()


def _raycast() -> None:
    pyautogui.hotkey("command", "space")


def _media_next() -> None:
    _keyboard.press(Key.media_next)
    _keyboard.release(Key.media_next)


def _media_prev() -> None:
    _keyboard.press(Key.media_previous)
    _keyboard.release(Key.media_previous)


def _media_play_pause() -> None:
    _keyboard.press(Key.media_play_pause)
    _keyboard.release(Key.media_play_pause)


def _volume_up() -> None:
    _keyboard.press(Key.media_volume_up)
    _keyboard.release(Key.media_volume_up)


HANDLERS: Dict[str, Callable[[], None]] = {
    "raycast": _raycast,
    "media_next": _media_next,
    "media_prev": _media_prev,
    "media_play_pause": _media_play_pause,
    "volume_up": _volume_up,
}


class ActionDispatcher(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self._cooldown: float = self.config.get("cooldown_s", 1.0)
        # gesture name -> action, fired once when the gesture starts (the
        # arbiter in gesture_hci decides when that is). on_point_action is the
        # legacy spelling of bindings.point ("" / None disables it).
        self._bindings: Dict[str, str] = {
            "point": self.config.get("on_point_action", "raycast"),
            **(self.config.get("bindings") or {}),
        }
        self._last_ts: float = -float("inf")
        self.bus.subscribe(ActionEvent, self._on_action)
        self.bus.subscribe(GestureEvent, self._on_gesture)

    def _fire(self, action_type: str) -> bool:
        """Run an action if past cooldown. Returns True if it fired."""
        if time.perf_counter() - self._last_ts < self._cooldown:
            return False
        handler = HANDLERS.get(action_type)
        if handler is None:
            logger.warning("Unknown action: %s", action_type)
            return False
        handler()
        self._last_ts = time.perf_counter()
        return True

    async def _on_action(self, event: ActionEvent) -> None:
        self._fire(event.action_type)

    async def _on_gesture(self, event: GestureEvent) -> None:
        action = self._bindings.get(event.name)
        if event.phase == "start" and action:
            self._fire(action)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(3600)
