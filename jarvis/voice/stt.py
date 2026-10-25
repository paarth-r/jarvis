"""Speech-to-text: mlx-whisper with vocabulary priming and a phantom-text guard."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import numpy as np

NO_SPEECH_THRESHOLD = 0.6
PHANTOMS = {
    "thank you for watching",
    "thanks for watching",
    "you",
    "thank you",
    "bye",
    "subtitles by the amaraorg community",
}
_WAKE_PREFIX = re.compile(r"^\s*hey[,\s]+(?:jarvis|claude)\b[,.]?\s*", re.IGNORECASE)


class STT(Protocol):
    def transcribe(self, audio: np.ndarray) -> str | None: ...


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def clean_transcript(result: dict) -> str | None:
    """Return stripped text, or None for empty, no-speech, or known phantom output."""
    text = (result.get("text") or "").strip()
    if not text:
        return None
    segments = result.get("segments") or []
    if segments and all(s.get("no_speech_prob", 0.0) > NO_SPEECH_THRESHOLD for s in segments):
        return None
    if _normalize(text) in PHANTOMS:
        return None
    return text


def strip_wake_phrase(text: str) -> str:
    """Remove a leading "hey jarvis" / "hey claude" from a transcript."""
    return _WAKE_PREFIX.sub("", text, count=1).strip()


def load_vocab_prompt(path: Path | None) -> str | None:
    """Build a Whisper initial prompt from a one-term-per-line file."""
    if path is None or not Path(path).is_file():
        return None
    terms = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not terms:
        return None
    return "Vocabulary: " + ", ".join(terms) + "."


class WhisperSTT:
    def __init__(self, model: str, initial_prompt: str | None = None) -> None:
        self.model = model
        self.initial_prompt = initial_prompt

    def transcribe(self, audio: np.ndarray) -> str | None:
        import mlx_whisper  # lazy: heavy import, Apple silicon only

        samples = audio.astype(np.float32) / 32768.0
        result = mlx_whisper.transcribe(
            samples,
            path_or_hf_repo=self.model,
            language="en",
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
        )
        return clean_transcript(result)
