import pytest

from jarvis.voice.mac_speakers import MacSpeakers, SpeakerError, to_spotify_uri
from jarvis.voice.room_stub import RoomState


class FakeMac:
    """Records AppleScripts and answers the few reads MacSpeakers makes."""

    def __init__(self, running=True, status="paused", volume=31):
        self.scripts, self.running, self.status, self.volume = [], running, status, volume

    def __call__(self, script):
        self.scripts.append(script)
        if script == 'application "Spotify" is running':
            return "true" if self.running else "false"
        if script.startswith("output volume"):
            return str(self.volume)
        if "player state" in script:
            return f"{self.status}\tClair de Lune by Debussy"
        if script.startswith("set volume output volume"):
            self.volume = int(script.rsplit(" ", 1)[1])
        return ""


def test_uri_parsing():
    assert to_spotify_uri("spotify:track:abc123") == "spotify:track:abc123"
    assert to_spotify_uri("https://open.spotify.com/intl-de/playlist/37i9dQ?si=x") == "spotify:playlist:37i9dQ"
    assert to_spotify_uri("clair de lune") is None


def test_state_reads_live():
    mac = FakeMac(status="playing")
    state = MacSpeakers(mac).state()
    assert state == {"playing": True, "query": "", "volume": 31,
                     "now_playing": "Clair de Lune by Debussy"}


def test_state_does_not_launch_spotify():
    mac = FakeMac(running=False)
    assert MacSpeakers(mac).state()["playing"] is False
    assert not any("player state" in s for s in mac.scripts)


def test_query_searches_then_plays_uri():
    mac = FakeMac()
    MacSpeakers(mac, search=lambda q, k: "spotify:track:XYZ").apply({"query": "clair de lune", "playing": True})
    plays = [s for s in mac.scripts if "play" in s and "player state" not in s]
    assert plays == ['tell application "Spotify" to play track "spotify:track:XYZ"']


def test_query_without_credentials_errors_but_links_work():
    mac = FakeMac()
    with pytest.raises(SpeakerError, match="not set up"):
        MacSpeakers(mac).apply({"query": "clair de lune"})
    MacSpeakers(mac).apply({"query": "spotify:album:A1"})
    assert mac.scripts[-1] == 'tell application "Spotify" to play track "spotify:album:A1"'


def test_pause_resume_and_volume():
    mac = FakeMac()
    sp = MacSpeakers(mac)
    sp.apply({"playing": False})
    assert mac.scripts[-1] == 'tell application "Spotify" to pause'
    sp.apply({"playing": True, "volume": 55})
    assert 'tell application "Spotify" to play' in mac.scripts
    assert mac.volume == 55


def test_room_uses_real_speakers_and_validates_first():
    mac = FakeMac()
    room = RoomState(MacSpeakers(mac))
    status, dev = room.patch("speakers", {"volume": 40})
    assert status == 200 and dev["state"]["volume"] == 40
    before = len(mac.scripts)
    status, body = room.patch("speakers", {"volume": 400, "playing": False})
    assert status == 400 and len(mac.scripts) == before  # nothing applied
    status, body = room.patch("speakers", {"query": "no creds"})
    assert status == 502 and "not set up" in body["error"]
    assert room.get("fan")["state"] == {"on": False, "speed": 1}  # others stay fake
