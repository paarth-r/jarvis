"""The studio API end to end: real server, real bus, synthetic hands."""
import asyncio
import json
import time

import numpy as np
import pytest

from jarvis.events import GestureEvent, StudioModeEvent, WorldPoseEvent
from jarvis.gestures.library import GestureLibrary
from jarvis.runtime.event_bus import EventBus
from jarvis.studio.module import StudioModule
from tests.hand_synth import FIST, OPEN, THREE, hand, rotation
from tests.test_gesture_actions import FakeBackend
from tests.test_studio_http import _request


@pytest.fixture
async def studio():
    bus = EventBus()
    mod = StudioModule("studio", {"port": 0})
    await mod.setup(bus)
    mod.runner.backend = FakeBackend()
    await mod.server.start()
    modes = []

    async def on_mode(e):
        modes.append(e.recording)

    bus.subscribe(StudioModeEvent, on_mode)
    mod.modes = modes
    yield mod
    await mod.server.stop()


async def api(mod, method, path, body=None):
    status, payload = await _request(mod.server.port, method, path, body)
    return status, json.loads(payload) if payload else None


async def _feed(mod, pose, seconds, seed=0):
    rng = np.random.default_rng(seed)
    R = rotation(*rng.uniform(-0.15, 0.15, 3))
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        await mod.bus.publish(WorldPoseEvent(
            landmarks_3d=np.zeros((21, 3)), timestamp=time.perf_counter(), confidence=1.0,
            hand_side="right", hand_shape=hand(pose, R=R, noise=0.002, rng=rng)))
        await asyncio.sleep(1 / 30)


async def _take(mod, pose, seed=0, kind="take"):
    req = asyncio.create_task(api(mod, "POST", "/api/candidate/record",
                                  {"kind": kind, "countdown_s": 0, "seconds": 0.3}))
    await _feed(mod, pose, 0.4, seed)
    return await req


async def test_lists_builtins_with_their_roles(studio):
    status, body = await api(studio, "GET", "/api/gestures")
    by = {g["name"]: g for g in body["gestures"]}
    assert status == 200 and set(by) == {"pinch", "fist", "point"}
    assert by["fist"]["reserved"] and by["fist"]["binding_text"] == "Scroll"
    assert by["point"]["binding_text"] == "Keys: cmd+space"


async def test_bind_and_unbind_point(studio):
    status, entry = await api(studio, "POST", "/api/gestures/point/binding",
                              {"action": {"type": "media", "key": "next"}})
    assert status == 200 and entry["binding_text"] == "Media: next track"
    assert GestureLibrary().bindings["point"] == {"type": "media", "key": "next"}


async def test_reserved_and_invalid_bindings_are_400_with_reason(studio):
    status, body = await api(studio, "POST", "/api/gestures/fist/binding",
                             {"action": {"type": "media", "key": "next"}})
    assert status == 400 and "reserved" in body["error"]
    status, body = await api(studio, "POST", "/api/gestures/point/binding",
                             {"action": {"type": "keys", "combo": "cmd+zz"}})
    assert status == 400 and "key name" in body["error"]


async def test_validate_and_test_an_action(studio):
    _, v = await api(studio, "POST", "/api/actions/validate", {"action": {"type": "keys", "combo": "cmd+c"}})
    assert v == {"errors": [], "text": "Keys: cmd+c"}
    status, _ = await api(studio, "POST", "/api/actions/test", {"action": {"type": "media", "key": "mute"}})
    assert status == 200 and studio.runner.backend.calls == [("media", "mute")]


async def test_bad_candidate_names(studio):
    for name in ("", "fist", "Bad/Name"):
        status, body = await api(studio, "POST", "/api/candidate", {"name": name})
        assert status == 400 and body["error"]


async def test_full_add_flow_records_assesses_saves_and_binds(studio):
    status, _ = await api(studio, "POST", "/api/candidate", {"name": "three"})
    assert status == 200
    for i in range(4):
        status, rec = await _take(studio, THREE, seed=i)
        assert status == 200 and rec["tracked"] > 0.8
    assert rec["takes"] == 4
    # Cursor/actions were paused for each take and resumed after.
    assert studio.modes == [True, False] * 4

    status, report = await api(studio, "POST", "/api/candidate/assess")
    assert status == 200 and report["verdict"] == "pass", report
    assert "samples" not in report

    status, entry = await api(studio, "POST", "/api/candidate/save",
                              {"action": {"type": "media", "key": "play_pause"}})
    assert status == 200 and entry["kind"] == "custom" and entry["verdict"] == "pass"
    lib = GestureLibrary()
    assert "three" in lib.gestures and lib.bindings["three"]["key"] == "play_pause"
    assert studio.candidate is None


async def test_failed_check_blocks_saving(studio):
    await api(studio, "POST", "/api/candidate", {"name": "squeeze"})
    for i in range(3):
        await _take(studio, FIST, seed=i)
    _, report = await api(studio, "POST", "/api/candidate/assess")
    assert report["verdict"] == "fail"
    assert "fist" in report["checks"]["distinct"]["summary"]
    status, body = await api(studio, "POST", "/api/candidate/save", {})
    assert status == 400 and "failed" in body["error"]


async def test_redo_a_take(studio):
    await api(studio, "POST", "/api/candidate", {"name": "three"})
    for i in range(3):
        await _take(studio, THREE, seed=i)
    status, body = await api(studio, "DELETE", "/api/candidate/takes/2")
    assert status == 200 and body["takes"] == 2
    status, _ = await api(studio, "DELETE", "/api/candidate/takes/9")
    assert status == 400


async def test_background_clip_feeds_the_distinct_check(studio):
    await api(studio, "POST", "/api/candidate", {"name": "three"})
    for i in range(3):
        await _take(studio, THREE, seed=i)
    _, rec = await _take(studio, THREE, seed=9, kind="background")
    assert rec["has_background"]
    _, report = await api(studio, "POST", "/api/candidate/assess")
    assert report["checks"]["distinct"]["details"]["background_hits"] > 0.5
    assert report["verdict"] == "fail"


async def test_only_one_recording_at_a_time(studio):
    await api(studio, "POST", "/api/candidate", {"name": "three"})
    first = asyncio.create_task(api(studio, "POST", "/api/candidate/record",
                                    {"countdown_s": 0, "seconds": 0.3}))
    await asyncio.sleep(0.05)
    status, body = await api(studio, "POST", "/api/candidate/record", {"countdown_s": 0, "seconds": 0.3})
    assert status == 400 and "already" in body["error"]
    await first


async def test_retrain_keeps_binding_and_skips_comparing_with_itself(studio):
    await api(studio, "POST", "/api/candidate", {"name": "three"})
    for i in range(3):
        await _take(studio, THREE, seed=i)
    await api(studio, "POST", "/api/candidate/assess")
    await api(studio, "POST", "/api/candidate/save", {"action": {"type": "media", "key": "next"}})

    status, _ = await api(studio, "POST", "/api/candidate", {"name": "three", "replace": True})
    assert status == 200
    for i in range(3):
        await _take(studio, THREE, seed=10 + i)
    _, report = await api(studio, "POST", "/api/candidate/assess")
    assert report["verdict"] == "pass"                     # not "too close to three"
    _, entry = await api(studio, "POST", "/api/candidate/save", {})
    assert entry["binding"] == {"type": "media", "key": "next"}


async def test_delete_custom_gesture(studio):
    GestureLibrary().add_gesture("three", np.zeros((3, 23)), 0.3)
    status, _ = await api(studio, "DELETE", "/api/gestures/three")
    assert status == 200 and "three" not in GestureLibrary().gestures
    status, _ = await api(studio, "DELETE", "/api/gestures/point")
    assert status == 400


async def test_state_has_hand_and_recent_gesture_events(studio):
    await studio.bus.publish(WorldPoseEvent(np.zeros((21, 3)), time.perf_counter(), 1.0, "right",
                                            hand_shape=hand(OPEN)))
    await studio.bus.publish(GestureEvent("point", "start", time.perf_counter()))
    _, st = await api(studio, "GET", "/api/state")
    assert np.asarray(st["hand"]).shape == (21, 3)
    assert st["events"][-1]["name"] == "point"


async def test_port_env_overrides_config(monkeypatch):
    monkeypatch.setenv("JARVIS_STUDIO_PORT", "0")
    mod = StudioModule("studio", {"port": 8765})
    await mod.setup(EventBus())
    await mod.server.start()
    try:
        assert mod.server.port not in (0, 8765)
    finally:
        await mod.server.stop()


async def test_index_and_static_files(studio):
    status, _ = await _request(studio.server.port, "GET", "/")
    assert status == 200
    status, _ = await _request(studio.server.port, "GET", "/static/..%2Fmodule.py")
    assert status == 404
