"""Tests for the jarvis-voice argument parser."""
from __future__ import annotations

from pathlib import Path

from jarvis.voice.main import build_parser


def test_defaults():
    args = build_parser().parse_args([])
    assert args.config == Path("config.voice.example.yaml")
    assert args.local == Path("config.voice.local.yaml")
    assert not (args.list_devices or args.no_room or args.check)


def test_flags():
    args = build_parser().parse_args(
        ["--config", "a.yaml", "--local", "b.yaml", "--no-room", "--check"])
    assert args.config == Path("a.yaml") and args.local == Path("b.yaml")
    assert args.no_room and args.check


async def test_close_frames_stops_the_mic_iterator():
    from types import SimpleNamespace

    from jarvis.voice.main import close_frames

    closed = []

    async def frames():
        try:
            while True:
                yield 0
        finally:
            closed.append(True)

    it = frames()
    await it.__anext__()
    await close_frames(SimpleNamespace(_frames=it))
    await close_frames(SimpleNamespace(_frames=None))
    assert closed == [True]
