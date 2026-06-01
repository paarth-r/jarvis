import asyncio
import pytest
from unittest.mock import MagicMock, patch
from jarvis.runtime.event_bus import EventBus
from jarvis.modules.action_dispatcher import ActionDispatcher
from jarvis.events import ActionEvent
import time


@pytest.mark.asyncio
async def test_unknown_action_does_not_raise():
    bus = EventBus()
    mod = ActionDispatcher("action_dispatcher", {})
    await mod.setup(bus)

    # Should log a warning, not raise
    await bus.publish(ActionEvent(action_type="nonexistent", params={}, timestamp=time.perf_counter()))


@pytest.mark.asyncio
async def test_raycast_action_calls_hotkey():
    bus = EventBus()
    mod = ActionDispatcher("action_dispatcher", {})
    await mod.setup(bus)

    with patch("jarvis.modules.action_dispatcher.pyautogui") as mock_pyautogui:
        await bus.publish(ActionEvent(
            action_type="raycast", params={}, timestamp=time.perf_counter()
        ))
        mock_pyautogui.hotkey.assert_called_once_with("command", "space")


@pytest.mark.asyncio
async def test_cooldown_prevents_rapid_repeat():
    bus = EventBus()
    mod = ActionDispatcher("action_dispatcher", {"cooldown_s": 1.0})
    await mod.setup(bus)

    call_count = [0]

    with patch("jarvis.modules.action_dispatcher.pyautogui") as mock_pyautogui:
        mock_pyautogui.hotkey.side_effect = lambda *a: call_count.__setitem__(0, call_count[0] + 1)
        ts = time.perf_counter()
        await bus.publish(ActionEvent(action_type="raycast", params={}, timestamp=ts))
        await bus.publish(ActionEvent(action_type="raycast", params={}, timestamp=ts + 0.1))

    assert call_count[0] == 1
