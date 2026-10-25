"""Voice log: one markdown line per turn in the vault, plus an utterance archive.

Turns go to ``<vault>/wiki/voice-log/YYYY-MM-DD.md``. Every recorded utterance
(including ones Whisper dropped and confirmation answers) is kept as a WAV under
``<data_dir>/utterances/`` with a row in ``<data_dir>/utterances.jsonl`` so STT
mistakes can be corrected and measured later. Neither location is in the repo.
"""
from __future__ import annotations

import json
import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class VoiceLog:
    def __init__(self, vault: Path, data_dir: Path,
                 clock: Callable[[], datetime] = datetime.now) -> None:
        self._dir = Path(vault) / "wiki" / "voice-log"
        self._data = Path(data_dir)
        self._clock = clock

    def log_turn(self, heard: str, said: str) -> None:
        """Append ``- HH:MM **heard:** ... / **said:** ...`` to today's file."""
        now = self._clock()
        day = now.strftime("%Y-%m-%d")
        path = self._dir / f"{day}.md"
        self._dir.mkdir(parents=True, exist_ok=True)
        header = "" if path.exists() else f"# Voice log {day}\n\n"
        line = (f"- {now:%H:%M} **heard:** {_one_line(heard)} / "
                f"**said:** {_one_line(said)}\n")
        with path.open("a", encoding="utf-8") as f:
            f.write(header + line)

    def archive_utterance(self, audio: np.ndarray, transcript: str | None) -> Path:
        """Save the utterance as 16 kHz mono WAV and append its jsonl row."""
        now = self._clock()
        out_dir = self._data / "utterances"
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = _unique(out_dir / f"{now:%Y-%m-%dT%H-%M-%S-%f}.wav")
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(_to_int16(audio).tobytes())
        row = {"ts": now.isoformat(), "wav": str(wav),
               "transcript": transcript, "corrected": None}
        with (self._data / "utterances.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return wav


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _unique(path: Path) -> Path:
    """Avoid clobbering an earlier WAV when two land on the same timestamp."""
    n = 1
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        n += 1
    return candidate


def _to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.floating):
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return audio.astype(np.int16)
