"""jarvis-dj: the Spotify command line any Claude session can drive."""
from __future__ import annotations

import pytest

from jarvis.voice.dj import run
from jarvis.voice.mac_speakers import MacSpeakers, SpeakerError
from jarvis.voice.spotify_user import UserLibrary
from tests.voice.test_mac_speakers import FakeMac
from tests.voice.test_spotify_user import FakeCatalog

TRACK = {"uri": "spotify:track:T1", "name": "Hype", "artists": [{"name": "Travis"}]}
ARTIST = {"uri": "spotify:artist:A1", "name": "Travis"}


class FakeAccount(FakeCatalog):
    def current_user_top_tracks(self, limit, time_range):
        self.top_args = (limit, time_range)
        return {"items": [TRACK, None]}

    def current_user_top_artists(self, limit, time_range):
        return {"items": [ARTIST]}

    def current_user_recently_played(self, limit):
        return {"items": [{"track": TRACK, "played_at": "2026-10-26T01:00:00Z"}]}


def make():
    mac, lib = FakeMac(status="playing"), UserLibrary(FakeAccount())
    return mac, MacSpeakers(mac, search=lib, library=lib), lib


def dj(argv, speakers, lib):
    return run(argv, speakers, lib, sleep=lambda s: None)


def test_play_a_track_reports_what_is_playing():
    mac, speakers, lib = make()
    out = dj(["play", "clair de lune"], speakers, lib)
    assert 'tell application "Spotify" to play track "spotify:track:Z"' in mac.scripts
    assert out["now_playing"] == "Clair de Lune by Debussy"


def test_play_kinds_and_links():
    mac, speakers, lib = make()
    dj(["play", "gym hype", "--kind", "my_playlist"], speakers, lib)
    assert 'tell application "Spotify" to play track "spotify:playlist:A"' in mac.scripts
    dj(["play", "--kind", "liked"], speakers, lib)
    assert 'tell application "Spotify" to play track "spotify:user:user123:collection"' in mac.scripts
    dj(["play", "https://open.spotify.com/album/X9"], speakers, lib)
    assert 'tell application "Spotify" to play track "spotify:album:X9"' in mac.scripts


def test_transport_volume_and_shuffle():
    mac, speakers, lib = make()
    for argv in (["pause"], ["resume"], ["next"], ["prev"], ["shuffle", "on"], ["volume", "40"]):
        dj(argv, speakers, lib)
    for script in ('tell application "Spotify" to pause', 'tell application "Spotify" to play',
'tell application "Spotify" to next track',
                   'tell application "Spotify" to previous track',
                   'tell application "Spotify" to set shuffling to true', "set volume output volume 40"):
        assert script in mac.scripts
    with pytest.raises(SystemExit):
        dj(["volume", "140"], speakers, lib)


def test_now_drops_the_internal_query_field():
    _, speakers, lib = make()
    assert dj(["now"], speakers, lib) == {"playing": True, "volume": 31,
                                          "now_playing": "Clair de Lune by Debussy"}


def test_search_playlists_top_and_recent():
    _, speakers, lib = make()
    assert dj(["search", "radiohead", "--kind", "artist"], speakers, lib)[0]["uri"] == "spotify:artist:Z"
    assert [p["name"] for p in dj(["playlists"], speakers, lib)] == ["Gym Hype", "Late Night Study"]
    top = dj(["top", "--range", "long", "--limit", "5"], speakers, lib)
    assert top == [{"uri": "spotify:track:T1", "name": "Hype", "by": "Travis", "kind": "track"}]
    assert lib._sp.top_args == (5, "long_term")
    assert dj(["top", "artists"], speakers, lib)[0]["name"] == "Travis"
    recent = dj(["recent"], speakers, lib)
    assert recent[0]["name"] == "Hype" and recent[0]["played_at"].startswith("2026-10-26")


def test_library_commands_without_login_say_how_to_fix_it():
    speakers = MacSpeakers(FakeMac())
    with pytest.raises(SpeakerError, match="jarvis-spotify-login"):
        dj(["top"], speakers, None)
