"""The real room server exposes only real devices."""
from __future__ import annotations

import json
import threading
import urllib.request

from jarvis.voice.mac_speakers import MacSpeakers, SpotifySearch
from jarvis.voice.room_server import REAL_DEVICES, build_search
from jarvis.voice.room_stub import RoomState, make_server
from tests.voice.test_mac_speakers import FakeMac


def test_only_real_devices_are_listed():
    room = RoomState(MacSpeakers(FakeMac()), only=REAL_DEVICES)
    assert [d["id"] for d in room.list()] == ["speakers"]
    assert room.get("fan") is None
    assert room.patch("fan", {"on": True})[0] == 404


def test_stub_still_has_everything():
    assert {d["id"] for d in RoomState().list()} == {"fan", "window", "thermostat", "speakers"}


def test_search_prefers_config_then_env(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    assert build_search(None, None) is None
    assert isinstance(build_search("id", "secret"), SpotifySearch)
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "id")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
    assert isinstance(build_search(None, None), SpotifySearch)


def test_http_volume_round_trip():
    mac = FakeMac(volume=20)
    server = make_server("127.0.0.1", 0, MacSpeakers(mac), REAL_DEVICES)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/devices") as r:
            assert [d["id"] for d in json.loads(r.read())] == ["speakers"]
        req = urllib.request.Request(f"{base}/devices/speakers", method="POST",
                                     data=json.dumps({"volume": 45}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            assert json.loads(r.read())["state"]["volume"] == 45
        assert mac.volume == 45
    finally:
        server.shutdown()
        server.server_close()
