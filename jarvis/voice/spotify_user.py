"""Your own Spotify library (playlists, Liked Songs) via a one-time user login.

`jarvis-spotify-login` opens the browser once (Authorization Code with PKCE, no
client secret); the token is cached in ~/.jarvis/spotify_user_token.json and
refreshed automatically after that. The room server then resolves
"my playlist <name>" and "liked songs" for the speakers.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

from jarvis.voice.mac_speakers import SpeakerError

REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPES = " ".join([
    "playlist-read-private", "playlist-read-collaborative", "user-library-read",
    "user-top-read", "user-read-recently-played",
])
TOKEN_PATH = Path.home() / ".jarvis" / "spotify_user_token.json"


def make_auth(client_id: str, token_path: Path = TOKEN_PATH, open_browser: bool = True):
    from spotipy.oauth2 import SpotifyPKCE

    token_path.parent.mkdir(parents=True, exist_ok=True)
    return SpotifyPKCE(client_id=client_id, redirect_uri=REDIRECT_URI, scope=SCOPES,
                       cache_path=str(token_path), open_browser=open_browser)


class UserLibrary:
    """Your playlists and Liked Songs. `client` is a spotipy.Spotify (or a fake)."""

    def __init__(self, client) -> None:
        self._sp = client
        self._me: str | None = None

    @classmethod
    def from_cache(cls, client_id: str | None, token_path: Path = TOKEN_PATH) -> "UserLibrary | None":
        """A library if a login token is cached, else None (never opens a browser)."""
        if not client_id or not token_path.exists():
            return None
        import spotipy

        return cls(spotipy.Spotify(auth_manager=make_auth(client_id, token_path, open_browser=False)))

    def _user_id(self) -> str:
        if self._me is None:
            self._me = self._sp.current_user()["id"]
        return self._me

    def playlists(self) -> list[dict]:
        out, page = [], self._sp.current_user_playlists(limit=50)
        while page:
            out += [p for p in page.get("items", []) if p]
            page = self._sp.next(page) if page.get("next") else None
        return out

    def search_playlists(self, query: str, limit: int = 5) -> list[dict]:
        """Your playlists whose names best match `query` (substring first, then fuzzy)."""
        q = query.strip().lower()
        mine = self.playlists()
        names = [p.get("name", "") for p in mine]
        hits = [p for p, n in zip(mine, names) if q and q in n.lower()]
        if not hits:
            close = difflib.get_close_matches(q, [n.lower() for n in names], n=limit, cutoff=0.7)
            hits = [p for c in close for p, n in zip(mine, names) if n.lower() == c]
        return [{"uri": p.get("uri", ""), "name": p.get("name", ""),
                 "by": (p.get("owner") or {}).get("display_name", ""), "kind": "my_playlist"}
                for p in hits[:limit]]

    def catalog_search(self, query: str, kind: str = "track", limit: int = 5) -> list[dict]:
        """Public catalog search with your login, so no client secret is needed."""
        from jarvis.voice.mac_speakers import _summarize

        from jarvis.voice.mac_speakers import SEARCH_PAGE

        res = self._sp.search(q=query, type=kind, limit=SEARCH_PAGE)
        items = (res.get(f"{kind}s") or {}).get("items", [])
        return [_summarize(i, kind) for i in items if i][:max(1, limit)]

    def __call__(self, query: str, kind: str = "track") -> str:
        """Acts as the catalog Search when no client secret is configured."""
        hits = self.catalog_search(query, kind, limit=1)
        if not hits:
            raise SpeakerError(f"no Spotify {kind} found for '{query}'")
        return hits[0]["uri"]

    def top(self, kind: str = "tracks", time_range: str = "short_term", limit: int = 10) -> list[dict]:
        """Your most played tracks or artists over short/medium/long_term."""
        from jarvis.voice.mac_speakers import _summarize

        fetch = (self._sp.current_user_top_artists if kind == "artists"
                 else self._sp.current_user_top_tracks)
        page = fetch(limit=limit, time_range=time_range)
        return [_summarize(i, kind.rstrip("s")) for i in page.get("items", []) if i]

    def recent(self, limit: int = 10) -> list[dict]:
        """Your recently played tracks, newest first."""
        from jarvis.voice.mac_speakers import _summarize

        page = self._sp.current_user_recently_played(limit=limit)
        return [{**_summarize(i["track"], "track"), "played_at": i.get("played_at", "")}
                for i in page.get("items", []) if i and i.get("track")]

    def liked_uri(self) -> str:
        return f"spotify:user:{self._user_id()}:collection"

    def resolve(self, query: str, kind: str) -> str:
        if kind == "liked":
            return self.liked_uri()
        hits = self.search_playlists(query, limit=1)
        if not hits:
            raise SpeakerError(f"none of your playlists matches '{query}'")
        return hits[0]["uri"]

    def search(self, query: str, kind: str, limit: int = 5) -> list[dict]:
        if kind == "liked":
            return [{"uri": self.liked_uri(), "name": "Liked Songs", "by": "you", "kind": "liked"}]
        return self.search_playlists(query, limit)


def main(argv: list[str] | None = None) -> None:
    """jarvis-spotify-login: one browser login, then the token refreshes itself."""
    from jarvis.voice.config import load_config
    from jarvis.voice.main import resolve_config_path

    parser = argparse.ArgumentParser(description="Log Jarvis in to your Spotify account once")
    parser.add_argument("--config", default="config.voice.example.yaml")
    parser.add_argument("--local", default="config.voice.local.yaml")
    args = parser.parse_args(argv)
    local = resolve_config_path(Path(args.local))
    cfg = load_config(resolve_config_path(Path(args.config)), local if local.exists() else None)
    if not cfg.spotify_client_id:
        sys.exit("set spotify_client_id in your local config first")
    import spotipy

    print(f"Opening Spotify in your browser; approve, and it redirects to {REDIRECT_URI}.",
          file=sys.stderr, flush=True)
    sp = spotipy.Spotify(auth_manager=make_auth(cfg.spotify_client_id))
    me = sp.current_user()
    count = len(UserLibrary(sp).playlists())
    print(f"Logged in as {me.get('display_name') or me['id']}; {count} playlists visible. "
          f"Token cached at {TOKEN_PATH}.", file=sys.stderr)


if __name__ == "__main__":
    main()
