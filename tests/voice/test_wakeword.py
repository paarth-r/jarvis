"""Wake word and mic-mute tests; no audio devices required."""
from __future__ import annotations

import shutil
import subprocess
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from jarvis.voice.audio import FRAME, MicStream, tone
from jarvis.voice.wakeword import WakeWord, _prebuilt_path


def _frames(samples: np.ndarray) -> list[np.ndarray]:
    n = len(samples) // FRAME
    return [samples[i * FRAME:(i + 1) * FRAME] for i in range(n)]


def _say(tmp_path: Path, text: str) -> np.ndarray:
    out = tmp_path / "x.wav"
    subprocess.run(["say", "-o", str(out), "--data-format=LEI16@16000", text], check=True)
    with wave.open(str(out)) as w:
        assert w.getframerate() == 16000
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    pad = np.zeros(16000, dtype=np.int16)
    return np.concatenate([pad, data, pad])


def test_dropin_models_discovered(tmp_path: Path) -> None:
    shutil.copy(_prebuilt_path("hey_jarvis"), tmp_path / "hey_claude.onnx")
    ww = WakeWord(tmp_path, 0.5)
    assert "hey_jarvis" in ww.names
    assert "hey_claude" in ww.names


def test_silence_never_fires(tmp_path: Path) -> None:
    ww = WakeWord(tmp_path, 0.5)
    zero = np.zeros(FRAME, dtype=np.int16)
    assert all(ww.process(zero) is None for _ in range(200))


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS say not available")
def test_fires_on_synthesized_phrase(tmp_path: Path) -> None:
    ww = WakeWord(tmp_path, 0.5)
    hits = [ww.process(f) for f in _frames(_say(tmp_path, "hey jarvis"))]
    assert "hey_jarvis" in hits


@pytest.mark.skipif(shutil.which("say") is None, reason="macOS say not available")
def test_reset_clears_state(tmp_path: Path) -> None:
    ww = WakeWord(tmp_path, 0.5)
    hits = [ww.process(f) for f in _frames(_say(tmp_path, "hey jarvis"))]
    assert "hey_jarvis" in hits
    ww.reset()
    zero = np.zeros(FRAME, dtype=np.int16)
    assert all(ww.process(zero) is None for _ in range(50))


async def test_mic_mute_drops_frames_and_tail() -> None:
    mic = MicStream()
    frame = np.ones(FRAME, dtype=np.int16)
    mic._enqueue(frame)
    with mic.mute(0.05):
        mic._enqueue(frame)
    mic._enqueue(frame)  # inside the tail
    assert mic._queue.qsize() == 1
    time.sleep(0.06)
    mic._enqueue(frame)
    assert mic._queue.qsize() == 2


def test_tone_shapes() -> None:
    for kind in ("wake", "working", "error"):
        samples, sr = tone(kind)  # type: ignore[arg-type]
        assert samples.dtype == np.int16 and sr > 0 and len(samples) > 0
