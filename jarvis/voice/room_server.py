"""The real room API (see docs/voice/room-api.md): only devices that exist.

Today that is `speakers`: this Mac's output volume plus the Spotify desktop app
(see mac_speakers.py). Devices without hardware behind them (fan, window,
thermostat) are left out so Claude never reports controlling something that
isn't there; `jarvis-room-stub` still fakes all four for testing.

Run with `jarvis-room [--port 8790] [--local config.voice.local.yaml]`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jarvis.voice.config import load_config
from jarvis.voice.mac_speakers import MacSpeakers, SpotifySearch
from jarvis.voice.room_stub import DEFAULT_PORT, make_server

REAL_DEVICES = {"speakers"}


def build_search(client_id: str | None, client_secret: str | None) -> SpotifySearch | None:
    """Config credentials first, then SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET."""
    if client_id and client_secret:
        return SpotifySearch(client_id, client_secret)
    return SpotifySearch.from_env()


def main(argv: list[str] | None = None) -> None:
    from jarvis.voice.main import resolve_config_path

    parser = argparse.ArgumentParser(description="Room API for the Jarvis voice agent")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--config", default="config.voice.example.yaml")
    parser.add_argument("--local", default="config.voice.local.yaml")
    args = parser.parse_args(argv)
    local = resolve_config_path(Path(args.local))
    cfg = load_config(resolve_config_path(Path(args.config)), local if local.exists() else None)
    search = build_search(cfg.spotify_client_id, cfg.spotify_client_secret)
    if search is None:
        print("[room] Spotify search not configured: only Spotify links can be played "
              "(set spotify_client_id / spotify_client_secret in your local config)",
              file=sys.stderr)
    server = make_server("127.0.0.1", args.port, MacSpeakers(search=search), REAL_DEVICES)
    print(f"[room] listening on http://127.0.0.1:{args.port} with {', '.join(sorted(REAL_DEVICES))}",
          file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
