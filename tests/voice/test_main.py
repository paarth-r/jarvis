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


def test_relative_paths_inside_the_config_fall_back_too(tmp_path, monkeypatch):
    from jarvis.voice.config import VoiceConfig
    from jarvis.voice.main import resolve_paths

    root = tmp_path / "repo"
    (root / "models").mkdir(parents=True)
    (root / "models" / "turn.onnx").write_bytes(b"")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = VoiceConfig(vault=Path("/abs/vault"), kokoro_model=Path("missing.onnx"),
                      kokoro_voices=Path("missing.bin"),
                      smart_turn_model=Path("models/turn.onnx"), vocab_file=None)
    out = resolve_paths(cfg, root)
    assert out.smart_turn_model == root / "models" / "turn.onnx"
    assert out.kokoro_model == Path("missing.onnx")  # found nowhere: left alone
    assert out.vault == Path("/abs/vault")
    assert out.vocab_file is None
