"""Voice-agent configuration: a frozen dataclass loaded from layered YAML."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PATH_FIELDS = {
    "vault", "wake_models_dir", "vocab_file", "kokoro_model", "kokoro_voices",
    "smart_turn_model", "data_dir",
}
DEFAULT_STOP_PHRASES = ("wait", "no", "stop", "hold up", "hang on", "hold on",
                        "nevermind", "never mind", "jarvis")


@dataclass(frozen=True, kw_only=True)
class VoiceConfig:
    vault: Path
    model: str = "claude-opus-5-5"
    wake_threshold: float = 0.5
    wake_models_dir: Path = Path("models/wakeword")
    stt_model: str = "mlx-community/whisper-large-v3-turbo"
    vocab_file: Path | None = Path("vocab.local.txt")
    self_chat_id: str | None = None
    tts_voice: str = "bm_daniel"
    address_as: str = "sir"
    permission_mode: str = "auto"
    spotify_client_id: str | None = None
    spotify_client_secret: str | None = None
    kokoro_model: Path
    kokoro_voices: Path
    smart_turn_model: Path
    room_api_url: str = "http://127.0.0.1:8790"
    followup_s: float = 8.0
    hard_silence_s: float = 3.0
    pause_check_s: float = 0.24
    max_utterance_s: float = 30.0
    working_tone_after_s: float = 3.0
    turn_timeout_s: float = 180.0
    idle_reset_s: float = 600.0
    duck_music: bool = True
    duck_ratio: float = 0.2
    barge_in: bool = True
    stop_phrases: tuple[str, ...] = DEFAULT_STOP_PHRASES
    data_dir: Path = Path("~/.jarvis/voice").expanduser()
    input_device: str | int | None = None
    output_device: str | int | None = None


def _read(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    return data or {}


def load_config(example: Path, local: Path | None) -> VoiceConfig:
    """Load the example YAML, overlay the local YAML (if it exists), validate."""
    raw = _read(example)
    if local is not None and Path(local).exists():
        raw.update(_read(Path(local)))
    known = {f.name for f in fields(VoiceConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown voice config key(s): {', '.join(unknown)}")
    for key in PATH_FIELDS & raw.keys():
        if raw[key] is not None:
            raw[key] = Path(raw[key]).expanduser()
    if raw.get("stop_phrases") is not None:
        # A bare no / yes in YAML is a boolean: map it back to the word.
        raw["stop_phrases"] = tuple(
            ("yes" if p else "no") if isinstance(p, bool) else str(p)
            for p in raw["stop_phrases"])
    return VoiceConfig(**raw)
