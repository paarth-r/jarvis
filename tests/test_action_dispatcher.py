import asyncio
import time

import numpy as np
import pytest

from jarvis.events import ActionEvent, GestureEvent, StudioModeEvent
from jarvis.gestures.library import GestureLibrary
from jarvis.modules.action_dispatcher import ActionDispatcher
from jarvis.runtime.event_bus import EventBus
from tests.test_gesture_actions import FakeBackend


async def _dispatcher(config=None):
    bus = EventBus()
    mod = ActionDispatcher("action_dispatcher", config or {})
    await mod.setup(bus)
    backend = FakeBackend()
    mod.runner.backend = backend
    return bus, mod, backend


def _start(name):
    return GestureEvent(name, "start", time.perf_counter())


def _end(name):
    return GestureEvent(name, "end", time.perf_counter())


@pytest.mark.asyncio
async def test_unknown_action_does_not_raise():
    bus, _, backend = await _dispatcher()
    await bus.publish(ActionEvent(action_type="nonexistent", params={}, timestamp=time.perf_counter()))
    assert backend.calls == []


@pytest.mark.asyncio
async def test_raycast_action_presses_cmd_space():
    bus, _, backend = await _dispatcher()
    await bus.publish(ActionEvent(action_type="raycast", params={}, timestamp=time.perf_counter()))
    assert backend.calls == [("hotkey", ("command", "space"))]


@pytest.mark.asyncio
async def test_point_gesture_start_fires_configured_action():
    bus, _, backend = await _dispatcher({"on_point_action": "raycast"})
    await bus.publish(_start("point"))
    assert backend.calls == [("hotkey", ("command", "space"))]


@pytest.mark.asyncio
async def test_point_gesture_disabled_when_no_action():
    bus, _, backend = await _dispatcher({"on_point_action": ""})
    await bus.publish(_start("point"))
    assert backend.calls == []


@pytest.mark.asyncio
async def test_cooldown_prevents_rapid_repeat():
    bus, _, backend = await _dispatcher({"cooldown_s": 1.0})
    await bus.publish(_start("point"))
    await bus.publish(_end("point"))
    await bus.publish(_start("point"))
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_cooldown_is_per_gesture():
    bus, mod, backend = await _dispatcher({"cooldown_s": 1.0, "bindings": {"peace": "media_next"}})
    mod.library.add_gesture("peace", np.zeros((3, 23)), radius=0.3)
    await bus.publish(_start("point"))
    await bus.publish(_start("peace"))
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_gesture_end_and_unbound_gestures_fire_nothing():
    bus, _, backend = await _dispatcher({"bindings": {"peace": "raycast"}})
    await bus.publish(_end("peace"))
    await bus.publish(_start("rock"))
    assert backend.calls == []
    await bus.publish(_start("peace"))
    assert backend.calls == [("hotkey", ("command", "space"))]


@pytest.mark.asyncio
async def test_library_bindings_override_config_and_reload_live():
    bus, mod, backend = await _dispatcher({"cooldown_s": 0.0})
    other_writer = GestureLibrary()           # the studio, in another object
    other_writer.set_binding("point", {"type": "media", "key": "play_pause"})
    await bus.publish(_start("point"))
    assert backend.calls == [("media", "play_pause")]
    other_writer.set_binding("point", {"type": "none"})   # explicitly off
    import os
    os.utime(other_writer.path, (2e9, 2e9))
    await bus.publish(_start("point"))
    assert backend.calls == [("media", "play_pause")]


@pytest.mark.asyncio
async def test_repeat_while_held_then_stops_on_end():
    bus, mod, backend = await _dispatcher()
    mod.library.set_binding("point", {"type": "media", "key": "volume_up", "repeat_s": 0.05})
    await bus.publish(_start("point"))
    await asyncio.sleep(0.18)
    await bus.publish(_end("point"))
    held = len(backend.calls)
    assert 3 <= held <= 5
    await asyncio.sleep(0.12)
    assert len(backend.calls) == held


@pytest.mark.asyncio
async def test_studio_recording_pauses_actions():
    bus, mod, backend = await _dispatcher()
    await bus.publish(StudioModeEvent(recording=True))
    await bus.publish(_start("point"))
    assert backend.calls == []
    await bus.publish(StudioModeEvent(recording=False))
    await bus.publish(_start("point"))
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_macro_runs_off_the_event_loop():
    bus, mod, backend = await _dispatcher()
    mod.library.set_binding("point", {"type": "macro", "steps": [
        {"type": "keys", "combo": "cmd+c"}, {"type": "wait", "seconds": 0.0},
        {"type": "media", "key": "next"}]})
    await bus.publish(_start("point"))
    for _ in range(50):
        if len(backend.calls) == 3:
            break
        await asyncio.sleep(0.01)
    assert backend.calls == [("hotkey", ("command", "c")), ("wait", 0.0), ("media", "next")]
