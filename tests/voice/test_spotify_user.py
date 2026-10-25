"""Your-library features, with spotipy replaced by a fake account."""
from __future__ import annotations

import pytest

from jarvis.voice.mac_speakers import MacSpeakers, SpeakerError
from jarvis.voice.room_server import REAL_DEVICES
from jarvis.voice.room_stub import RoomState
from jarvis.voice.spotify_user import UserLibrary
from tests.voice.test_mac_speakers import FakeMac


class FakeSpotify:
    def __init__(self):
        self.pages = [
            {"items": [{"uri": "spotify:playlist:A", "name": "Gym Hype", "owner": {"display_name": "me"}},
                       None], "next": "p2"},
            {"items": [{"uri": "spotify:playlist:B", "name": "Late Night Study", "owner": {"display_name": "me"}}],
             "next": None},
        ]

    def current_user(self):
        return {"id": "user123"}

    def current_user_playlists(self, limit=50):
        return self.pages[0]

    def next(self, page):
        return self.pages[1]


def test_lists_all_pages_and_matches_names():
    lib = UserLibrary(FakeSpotify())
    assert [p["name"] for p in lib.playlists()] == ["Gym Hype", "Late Night Study"]
    assert lib.search_playlists("study")[0]["uri"] == "spotify:playlist:B"
    assert lib.search_playlists("gym hipe")[0]["uri"] == "spotify:playlist:A"  # fuzzy
    assert lib.liked_uri() == "spotify:user:user123:collection"


def test_speakers_play_my_playlist_and_liked_songs():
    mac = FakeMac()
    room = RoomState(MacSpeakers(mac, library=UserLibrary(FakeSpotify())), only=REAL_DEVICES)
    assert room.patch("speakers", {"query": "gym hype", "kind": "my_playlist"})[0] == 200
    assert mac.scripts[-1] == 'tell application "Spotify" to play track "spotify:playlist:A"' or \
        'tell application "Spotify" to play track "spotify:playlist:A"' in mac.scripts
    assert room.patch("speakers", {"query": "my liked songs", "kind": "liked"})[0] == 200
    assert 'tell application "Spotify" to play track "spotify:user:user123:collection"' in mac.scripts
    status, results = room.search("", "liked", 5)
    assert status == 200 and results[0]["name"] == "Liked Songs"


def test_without_login_the_error_says_how_to_fix_it():
    room = RoomState(MacSpeakers(FakeMac()), only=REAL_DEVICES)
    status, body = room.patch("speakers", {"query": "gym", "kind": "my_playlist"})
    assert status == 502 and "jarvis-spotify-login" in body["error"]


def test_no_match_is_a_clear_error():
    with pytest.raises(SpeakerError, match="none of your playlists"):
        UserLibrary(FakeSpotify()).resolve("zzzz qqqq xxxx", "my_playlist")


def test_from_cache_needs_a_token(tmp_path):
    assert UserLibrary.from_cache("cid", tmp_path / "missing.json") is None
    assert UserLibrary.from_cache(None, tmp_path / "missing.json") is None


class FakeCatalog(FakeSpotify):
    def search(self, q, type, limit):
        return {f"{type}s": {"items": [{"uri": f"spotify:{type}:Z", "name": q.title(),
                                        "artists": [{"name": "Someone"}]}, None]}}


def test_login_doubles_as_catalog_search():
    mac, lib = FakeMac(), UserLibrary(FakeCatalog())
    room = RoomState(MacSpeakers(mac, search=lib, library=lib), only=REAL_DEVICES)
    assert room.patch("speakers", {"query": "clair de lune"})[0] == 200
    assert 'tell application "Spotify" to play track "spotify:track:Z"' in mac.scripts
    status, hits = room.search("radiohead", "artist", 3)
    assert status == 200 and hits[0]["uri"] == "spotify:artist:Z"


def test_unrelated_name_is_not_a_fuzzy_match():
    assert UserLibrary(FakeSpotify()).search_playlists("workout") == []
