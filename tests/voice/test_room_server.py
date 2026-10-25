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


class FakeSearch:
    """Spotify search double: callable for play, .search for listings."""

    def __init__(self):
        self.calls = []

    def __call__(self, query, kind="track"):
        self.calls.append((query, kind))
        return f"spotify:{kind}:X1"

    def search(self, query, kind="track", limit=5):
        self.calls.append((query, kind, limit))
        return [{"uri": f"spotify:{kind}:X1", "name": "Lofi Beats", "by": "Spotify", "kind": kind}]


def test_play_a_playlist_by_name_then_skip_and_shuffle():
    mac, search = FakeMac(), FakeSearch()
    room = RoomState(MacSpeakers(mac, search=search), only=REAL_DEVICES)
    status, _ = room.patch("speakers", {"query": "lofi beats", "kind": "playlist"})
    assert status == 200 and search.calls == [("lofi beats", "playlist")]
    assert 'tell application "Spotify" to play track "spotify:playlist:X1"' in mac.scripts
    room.patch("speakers", {"skip": "next", "shuffle": True})
    assert 'tell application "Spotify" to next track' in mac.scripts
    assert 'tell application "Spotify" to set shuffling to true' in mac.scripts
    assert room.patch("speakers", {"kind": "podcast", "query": "x"})[0] == 400
    assert room.patch("speakers", {"skip": "sideways"})[0] == 400


def test_search_endpoint_and_tool():
    import asyncio

    from jarvis.voice.room import RoomClient, make_room_tools

    server = make_server("127.0.0.1", 0, MacSpeakers(FakeMac(), search=FakeSearch()), REAL_DEVICES)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{base}/search?q=lofi&kind=playlist") as r:
            assert json.loads(r.read())[0]["uri"] == "spotify:playlist:X1"
        tool = {t.name: t for t in make_room_tools(RoomClient(base))}["search_music"]
        out = asyncio.run(tool.handler({"query": "radiohead", "kind": "artist"}))
        assert "spotify:artist:X1" in out["content"][0]["text"] and not out.get("is_error")
        bad = asyncio.run(tool.handler({"query": "x", "kind": "podcast"}))
        assert bad.get("is_error")
    finally:
        server.shutdown()
        server.server_close()


def test_search_without_credentials_is_a_clear_error():
    room = RoomState(MacSpeakers(FakeMac()), only=REAL_DEVICES)
    status, body = room.search("lofi", "playlist", 5)
    assert status == 502 and "not set up" in body["error"]
