"""jarvis-dj: Spotify on this Mac from the command line, for any Claude session.

The same speakers and library the room server uses (mac_speakers.py,
spotify_user.py), without the server: every command prints JSON. Library
commands (my_playlist, liked, playlists, top, recent) need the one-time
`jarvis-spotify-login`.

    jarvis-dj play "clair de lune"            jarvis-dj top --range short
    jarvis-dj play "gym" --kind my_playlist   jarvis-dj recent
    jarvis-dj play --kind liked               jarvis-dj search "radiohead" --kind artist
    jarvis-dj pause | resume | next | prev    jarvis-dj shuffle on
    jarvis-dj volume 40                       jarvis-dj now
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

from jarvis.voice.mac_speakers import ALL_KINDS, MacSpeakers, SpeakerError

RANGES = {"short": "short_term", "medium": "medium_term", "long": "long_term"}
# Spotify needs a moment to load a new context before it reports the track.
SETTLE_S = 1.5


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jarvis-dj", description="Spotify on this Mac, as JSON")
    p.add_argument("--local", default="config.voice.local.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    play = sub.add_parser("play", help="play a name or Spotify link")
    play.add_argument("query", nargs="?", default="")
    play.add_argument("--kind", choices=ALL_KINDS, default="track")
    search = sub.add_parser("search", help="look up without playing")
    search.add_argument("query")
    search.add_argument("--kind", choices=ALL_KINDS, default="track")
    search.add_argument("--limit", type=int, default=5)
    for name in ("now", "pause", "resume", "next", "prev", "playlists"):
        sub.add_parser(name)
    shuffle = sub.add_parser("shuffle")
    shuffle.add_argument("state", choices=("on", "off"))
    volume = sub.add_parser("volume")
    volume.add_argument("level", type=int, choices=range(0, 101), metavar="0-100")
    top = sub.add_parser("top", help="your most played")
    top.add_argument("what", nargs="?", choices=("tracks", "artists"), default="tracks")
    top.add_argument("--range", choices=RANGES, default="short")
    top.add_argument("--limit", type=int, default=10)
    recent = sub.add_parser("recent", help="your recently played")
    recent.add_argument("--limit", type=int, default=10)
    return p


def _need(library):
    if library is None:
        raise SpeakerError("this needs the one-time Spotify login: run jarvis-spotify-login")
    return library


def run(argv: list[str], speakers: MacSpeakers, library,
        sleep: Callable[[float], None] = time.sleep):
    """Run one command; returns what to print as JSON."""
    a = _parser().parse_args(argv)

    def now() -> dict:
        state = speakers.state()
        state.pop("query", None)
        return state

    if a.cmd == "play":
        if not a.query and a.kind != "liked":
            raise SpeakerError("play needs a name or Spotify link (or --kind liked)")
        speakers.apply({"query": a.query or a.kind, "kind": a.kind, "playing": True})
        sleep(SETTLE_S)
        return now()
    if a.cmd == "search":
        return speakers.search(a.query, a.kind, a.limit)
    if a.cmd == "playlists":
        return [{"uri": p.get("uri", ""), "name": p.get("name", ""),
                 "tracks": (p.get("tracks") or {}).get("total")}
                for p in _need(library).playlists()]
    if a.cmd == "top":
        return _need(library).top(a.what, RANGES[a.range], a.limit)
    if a.cmd == "recent":
        return _need(library).recent(a.limit)
    patch = {"pause": {"playing": False}, "resume": {"playing": True},
             "next": {"skip": "next"}, "prev": {"skip": "previous"}}.get(a.cmd)
    if a.cmd == "shuffle":
        patch = {"shuffle": a.state == "on"}
    elif a.cmd == "volume":
        patch = {"volume": a.level}
    if patch:
        speakers.apply(patch)
        if "skip" in patch:
            sleep(SETTLE_S)
    return now()


def main(argv: list[str] | None = None) -> None:
    from jarvis.voice.config import load_config
    from jarvis.voice.main import resolve_config_path
    from jarvis.voice.room_server import build_search
    from jarvis.voice.spotify_user import UserLibrary

    argv = sys.argv[1:] if argv is None else argv
    local = resolve_config_path(Path(_parser().parse_known_args(argv)[0].local))
    cfg = load_config(resolve_config_path(Path("config.voice.example.yaml")),
                      local if local.exists() else None)
    library = UserLibrary.from_cache(cfg.spotify_client_id)
    search = build_search(cfg.spotify_client_id, cfg.spotify_client_secret) or library
    try:
        out = run(argv, MacSpeakers(search=search, library=library), library)
    except SpeakerError as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
