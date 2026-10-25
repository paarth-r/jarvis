"""Real `speakers` device for the room API, backed by this Mac.

Volume is the macOS system output volume, so it drives whatever output is
current (e.g. a Bluetooth soundbar). Playback goes through the Spotify desktop
app over AppleScript. Spotify's AppleScript cannot play a search, so a free-text
`query` is resolved to a track through the Spotify Web API (client-credentials
flow, needs SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET); a Spotify URI or
open.spotify.com link plays without credentials.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
from typing import Callable

# runs one AppleScript, returns stdout stripped; raises SpeakerError on failure
Osascript = Callable[[str], str]
# resolves a free-text query of a kind (track/album/playlist/artist) to a spotify: URI
Search = Callable[[str, str], str]
KINDS = ("track", "album", "playlist", "artist")
# Spotify blanks most playlist hits for new developer apps (curated playlists come
# back as nulls), so always fetch a full page and keep the real ones.
SEARCH_PAGE = 10
# need the one-time user login (spotify_user.py): your playlists, your Liked Songs
USER_KINDS = ("my_playlist", "liked")
ALL_KINDS = KINDS + USER_KINDS


class SpeakerError(Exception):
    """A speaker command failed; the message is shown to Claude."""


def run_osascript(script: str) -> str:
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True,
                              timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SpeakerError(f"osascript failed: {exc}") from exc
    if proc.returncode != 0:
        raise SpeakerError(proc.stderr.strip() or "osascript failed")
    return proc.stdout.strip()


_URL = re.compile(r"open\.spotify\.com/(?:intl-[\w-]+/)?(track|album|playlist|artist|episode|show)/(\w+)")
_URI = re.compile(r"^spotify:(track|album|playlist|artist|episode|show):\w+$")


def to_spotify_uri(query: str) -> str | None:
    """A spotify: URI for a URI or open.spotify.com link, else None."""
    q = query.strip()
    if _URI.match(q):
        return q
    m = _URL.search(q)
    return f"spotify:{m.group(1)}:{m.group(2)}" if m else None


class SpotifySearch:
    """Top-track search via the Spotify Web API, client-credentials token cached."""

    def __init__(self, client_id: str, client_secret: str):
        self._auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        self._token: str | None = None
        self._expires = 0.0

    @classmethod
    def from_env(cls) -> "SpotifySearch | None":
        cid, secret = os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")
        return cls(cid, secret) if cid and secret else None

    def _get_token(self) -> str:
        if self._token and time.time() < self._expires - 60:
            return self._token
        req = urllib.request.Request(
            "https://accounts.spotify.com/api/token",
            data=b"grant_type=client_credentials",
            headers={"Authorization": f"Basic {self._auth}",
                     "Content-Type": "application/x-www-form-urlencoded"})
        body = self._fetch(req)
        self._token, self._expires = body["access_token"], time.time() + body["expires_in"]
        return self._token

    @staticmethod
    def _fetch(req: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            raise SpeakerError(f"Spotify search failed: {exc}") from exc

    def search(self, query: str, kind: str = "track", limit: int = 5) -> list[dict]:
        """Top results of one kind as {uri, name, by, kind}."""
        if kind not in KINDS:
            raise SpeakerError(f"kind must be one of {', '.join(KINDS)}")
        url = "https://api.spotify.com/v1/search?" + urllib.parse.urlencode(
            {"q": query, "type": kind, "limit": SEARCH_PAGE})
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._get_token()}"})
        items = self._fetch(req).get(f"{kind}s", {}).get("items", [])
        return [_summarize(i, kind) for i in items if i][:max(1, limit)]

    def __call__(self, query: str, kind: str = "track") -> str:
        results = self.search(query, kind, limit=1)
        if not results:
            raise SpeakerError(f"no Spotify {kind} found for '{query}'")
        return results[0]["uri"]


def _summarize(item: dict, kind: str) -> dict:
    if kind in ("track", "album"):
        by = ", ".join(a.get("name", "") for a in item.get("artists", []))
    elif kind == "playlist":
        by = (item.get("owner") or {}).get("display_name", "")
    else:
        by = ""
    return {"uri": item.get("uri", ""), "name": item.get("name", ""), "by": by, "kind": kind}


# the track lookup fails when Spotify has nothing loaded, so it is guarded
_NOW_PLAYING = """tell application "Spotify"
  set info to (player state as string) & tab
  try
    set info to info & (name of current track) & " by " & (artist of current track)
  end try
  return info
end tell"""


class MacSpeakers:
    """Reads live state each call; `query` is whatever was last asked for."""

    def __init__(self, osascript: Osascript = run_osascript, search: Search | None = None,
                 library=None):
        self._osa = osascript
        self._search = search
        self._library = library  # spotify_user.UserLibrary, or None before the login
        self._query = ""

    def _spotify_running(self) -> bool:
        return self._osa('application "Spotify" is running') == "true"

    def state(self) -> dict:
        volume = int(self._osa("output volume of (get volume settings)") or 0)
        playing, track = False, ""
        if self._spotify_running():
            info = self._osa(_NOW_PLAYING)
            status, _, track = info.partition("\t")
            playing = status == "playing"
        return {"playing": playing, "query": self._query, "volume": volume, "now_playing": track}

    def _need_library(self):
        if self._library is None:
            raise SpeakerError("your playlists and Liked Songs need the one-time Spotify "
                               "login: run jarvis-spotify-login")
        return self._library

    def _resolve(self, query: str, kind: str = "track") -> str:
        uri = to_spotify_uri(query)
        if uri:
            return uri
        if kind in USER_KINDS:
            return self._need_library().resolve(query, kind)
        if self._search is None:
            raise SpeakerError("Spotify search is not set up (SPOTIFY_CLIENT_ID/SECRET); "
                               "only Spotify links can be played")
        return self._search(query, kind)

    def search(self, query: str, kind: str = "track", limit: int = 5) -> list[dict]:
        if kind in USER_KINDS:
            return self._need_library().search(query, kind, limit)
        if self._search is None:
            raise SpeakerError("Spotify search is not set up (SPOTIFY_CLIENT_ID/SECRET)")
        find = getattr(self._search, "catalog_search", None) or getattr(self._search, "search", None)
        if find is None:
            raise SpeakerError("Spotify search is not set up (SPOTIFY_CLIENT_ID/SECRET)")
        return find(query, kind, limit)

    def apply(self, patch: dict) -> None:
        """Apply an already-validated patch: query, skip, shuffle, playing, volume.

        `query` is a song/album/playlist/artist name (pick which with `kind`,
        default track) or a Spotify link; albums, playlists and artists play as
        a whole.
        """
        if "query" in patch and patch["query"].strip():
            uri = self._resolve(patch["query"], patch.get("kind", "track"))
            self._osa(f'tell application "Spotify" to play track "{uri}"')
            self._query = patch["query"]
        if "skip" in patch:
            verb = "next track" if patch["skip"] == "next" else "previous track"
            self._osa(f'tell application "Spotify" to {verb}')
        if "shuffle" in patch:
            self._osa(f'tell application "Spotify" to set shuffling to {str(patch["shuffle"]).lower()}')
        if "playing" in patch:
            if patch["playing"]:
                if "query" not in patch:
                    self._osa('tell application "Spotify" to play')
            elif self._spotify_running():
                self._osa('tell application "Spotify" to pause')
        if "volume" in patch:
            self._osa(f"set volume output volume {patch['volume']}")
