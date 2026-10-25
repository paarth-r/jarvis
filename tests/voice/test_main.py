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


def test_config_paths_fall_back_to_the_repo_root(tmp_path, monkeypatch):
    from jarvis.voice.main import REPO_ROOT, resolve_config_path

    monkeypatch.chdir(tmp_path)  # a manual run from outside the repo
    example = resolve_config_path(Path("config.voice.example.yaml"))
    assert example == REPO_ROOT / "config.voice.example.yaml"
    assert example.exists()


def test_config_paths_prefer_the_working_directory(tmp_path, monkeypatch):
    from jarvis.voice.main import resolve_config_path

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.voice.example.yaml").write_text("vault: x\n")
    assert resolve_config_path(Path("config.voice.example.yaml")) == Path(
        "config.voice.example.yaml")
    # Absolute paths and paths found nowhere are left alone.
    assert resolve_config_path(tmp_path / "nope.yaml") == tmp_path / "nope.yaml"
    assert resolve_config_path(Path("missing.yaml")) == Path("missing.yaml")
