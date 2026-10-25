import json
import threading

import pytest

from jarvis.voice.room import RoomClient, RoomError, build_room_server, make_room_tools
from jarvis.voice.room_stub import make_server


@pytest.fixture
def base_url():
    server = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _tools(base_url):
    return {t.name: t for t in make_room_tools(RoomClient(base_url))}


async def test_list_devices(base_url):
    devices = await RoomClient(base_url).list_devices()
    assert {d["id"] for d in devices} == {"fan", "window", "thermostat", "speakers"}
    assert all({"id", "name", "kind", "state"} <= d.keys() for d in devices)


async def test_set_fan_on(base_url):
    client = RoomClient(base_url)
    device = await client.set_state("fan", {"on": True, "speed": 2})
    assert device["state"] == {"on": True, "speed": 2}
    assert (await client.get_state("fan"))["state"]["on"] is True


async def test_bad_patch_400(base_url):
    with pytest.raises(RoomError) as exc:
        await RoomClient(base_url).set_state("fan", {"speed": 9})
    assert exc.value.status == 400
    assert "speed" in str(exc.value)


async def test_unknown_404(base_url):
    with pytest.raises(RoomError) as exc:
        await RoomClient(base_url).get_state("toaster")
    assert exc.value.status == 404
    assert "unknown device" in str(exc.value)


async def test_tool_error_is_result(base_url):
    result = await _tools(base_url)["set_state"].handler({"device": "toaster", "patch": {"on": True}})
    assert result["is_error"] is True
    assert "unknown device" in result["content"][0]["text"]


async def test_tool_success_and_unreachable(base_url):
    tools = _tools(base_url)
    listed = await tools["list_devices"].handler({})
    assert len(json.loads(listed["content"][0]["text"])) == 4
    ok = await tools["set_state"].handler({"device": "window", "patch": {"open_pct": 40}})
    assert not ok.get("is_error")
    got = await tools["get_state"].handler({"device": "window"})
    assert json.loads(got["content"][0]["text"])["state"]["open_pct"] == 40
    dead = _tools("http://127.0.0.1:1")
    result = await dead["list_devices"].handler({})
    assert result["is_error"] is True


def test_build_room_server():
    cfg = build_room_server("http://127.0.0.1:8790")
    assert cfg["type"] == "sdk" and cfg["name"] == "room"


async def test_missing_args_are_error_results(base_url):
    tools = _tools(base_url)
    for name, args in [("get_state", {}), ("set_state", {}), ("set_state", {"device": "fan"}),
                       ("set_state", {"patch": {"on": True}})]:
        result = await tools[name].handler(args)
        assert result["is_error"] is True, (name, args)


async def test_empty_and_slashed_ids_do_not_change_path(base_url):
    client = RoomClient(base_url)
    with pytest.raises(RoomError, match="device id is required"):
        await client.get_state("")
    with pytest.raises(RoomError) as exc:
        await client.get_state("../devices")
    assert exc.value.status == 404
