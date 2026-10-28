"""Tests for the voice-agent config loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.voice.config import load_config

BASE = """\
vault: ~/path/to/vault
followup_s: 8.0
kokoro_model: models/kokoro.onnx
kokoro_voices: models/voices.bin
smart_turn_model: models/smart-turn.onnx
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_local_overrides_example(tmp_path: Path) -> None:
    ex = _write(tmp_path / "ex.yaml", BASE)
    local = _write(tmp_path / "local.yaml", "followup_s: 5.0\n")
    cfg = load_config(ex, local)
    assert cfg.followup_s == 5.0
    assert cfg.kokoro_model == Path("models/kokoro.onnx")
    assert cfg.hard_silence_s == 3.0
    assert cfg.pause_check_s == 0.24


def test_missing_local_uses_example(tmp_path: Path) -> None:
    ex = _write(tmp_path / "ex.yaml", BASE)
    assert load_config(ex, None).followup_s == 8.0
    assert load_config(ex, tmp_path / "absent.yaml").followup_s == 8.0


def test_unknown_key_raises(tmp_path: Path) -> None:
    ex = _write(tmp_path / "ex.yaml", BASE)
    local = _write(tmp_path / "local.yaml", "folowup_s: 5.0\n")
    with pytest.raises(ValueError, match="folowup_s"):
        load_config(ex, local)


def test_paths_expand_user(tmp_path: Path) -> None:
    ex = _write(tmp_path / "ex.yaml", BASE.replace("~/path/to/vault", "~/x"))
    cfg = load_config(ex, None)
    assert cfg.vault == Path.home() / "x"
    assert cfg.data_dir == Path.home() / ".jarvis" / "voice"


def test_shipped_example_defaults_to_daniel_and_sir():
    cfg = load_config(Path(__file__).resolve().parents[2] / "config.voice.example.yaml", None)
    assert cfg.tts_voice == "bm_daniel"
    assert cfg.address_as == "sir"


def test_duck_defaults():
    cfg = load_config(Path(__file__).resolve().parents[2] / "config.voice.example.yaml", None)
    assert cfg.duck_music is True
    assert cfg.duck_ratio == 0.2


def test_barge_in_defaults():
    cfg = load_config(Path(__file__).resolve().parents[2] / "config.voice.example.yaml", None)
    assert cfg.barge_in is True
    assert cfg.stop_phrases == ("wait", "no", "stop", "hold up", "hang on", "hold on",
                                "nevermind", "never mind", "jarvis")


def test_stop_phrases_yaml_list_becomes_a_tuple(tmp_path: Path) -> None:
    ex = _write(tmp_path / "ex.yaml", BASE + "stop_phrases: [wait, hold up]\nbarge_in: false\n")
    cfg = load_config(ex, None)
    assert cfg.stop_phrases == ("wait", "hold up")
    assert cfg.barge_in is False
