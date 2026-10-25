"""Tests for the voice log (daily markdown turns + utterance archive)."""
from __future__ import annotations

import json
import wave
from datetime import datetime

import numpy as np

from jarvis.voice.voicelog import VoiceLog


def _clock():
    return datetime(2026, 10, 25, 9, 7, 3, 120000)


def test_daily_file_created_with_header(tmp_path):
    log = VoiceLog(tmp_path / "vault", tmp_path / "data", clock=_clock)
    log.log_turn("what's due", "Report is due.")
    log.log_turn("thanks", "Anytime.\nBye.")
    path = tmp_path / "vault/wiki/voice-log/2026-10-25.md"
    lines = path.read_text().splitlines()
    assert lines[0] == "# Voice log 2026-10-25"
    assert "- 09:07 **heard:** what's due / **said:** Report is due." in lines
    assert "- 09:07 **heard:** thanks / **said:** Anytime. Bye." in lines
    assert sum(line.startswith("# Voice log") for line in lines) == 1


def test_archive_appends_jsonl(tmp_path):
    log = VoiceLog(tmp_path / "vault", tmp_path / "data", clock=_clock)
    audio = np.zeros(1600, dtype=np.int16)
    first = log.archive_utterance(audio, "what's due")
    log.archive_utterance(audio, None)
    assert first.parent == tmp_path / "data/utterances"
    with wave.open(str(first)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 1600)
    rows = [json.loads(line) for line in
            (tmp_path / "data/utterances.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["transcript"] == "what's due"
    assert rows[1]["transcript"] is None
    assert rows[0]["corrected"] is None
    assert rows[0]["wav"] == str(first)
    assert set(rows[0]) == {"ts", "wav", "transcript", "corrected"}
    assert rows[0]["wav"] != rows[1]["wav"]
