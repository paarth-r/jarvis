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
    frames = _frames(_say(tmp_path, "hey jarvis"))
    probe = WakeWord(tmp_path, 0.5)
    first = next(i for i, f in enumerate(frames) if probe.process(f))
    rest = frames[first + 1:]

    # Control: without reset the continuation of the phrase keeps firing.
    ww = WakeWord(tmp_path, 0.5)
    for f in frames[:first + 1]:
        ww.process(f)
    assert any(ww.process(f) for f in rest)

    # With reset the buffered audio is gone, so the same continuation is silent.
    ww = WakeWord(tmp_path, 0.5)
    for f in frames[:first + 1]:
        ww.process(f)
    ww.reset()
    assert not any(ww.process(f) for f in rest)


async def test_mic_mute_drops_frames_and_tail() -> None:
    mic = MicStream()
    frame = np.ones(FRAME, dtype=np.int16)
    mic._enqueue(frame)
    with mic.mute(0.05):
        mic._enqueue(frame)
    mic._enqueue(frame)  # inside the tail
    assert mic._queue.qsize() == 0
    time.sleep(0.06)
    mic._enqueue(frame)
    assert mic._queue.qsize() == 1


def test_tone_shapes() -> None:
    for kind in ("wake", "working", "error"):
        samples, sr = tone(kind)  # type: ignore[arg-type]
        assert samples.dtype == np.int16 and sr > 0 and len(samples) > 0


async def test_mic_mute_drains_queued_frames() -> None:
    mic = MicStream()
    frame = np.ones(FRAME, dtype=np.int16)
    mic._enqueue(frame)
    mic._enqueue(frame)
    with mic.mute(0.0):
        assert mic._queue.qsize() == 0
        mic._enqueue(frame)
    assert mic._queue.qsize() == 0


class FakeStream:
    """Stands in for sd.OutputStream; the test drives the callback by hand."""

    instances: list = []

    def __init__(self, samplerate, channels, dtype, device, callback):
        self.samplerate, self.callback, self.latency = samplerate, callback, 0.0
        self.started = self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True

    def pull(self, frames):
        out = np.ones((frames, 1), np.float32)
        self.callback(out, frames, None, None)
        return out[:, 0]


async def _wait_until(cond):
    import asyncio
    for _ in range(500):
        if cond():
            return
        await asyncio.sleep(0.002)
    raise AssertionError("condition never became true")


async def test_player_plays_whole_clip_then_silence() -> None:
    import asyncio
    from jarvis.voice import audio

    FakeStream.instances = []
    player = audio.Player(stream_factory=FakeStream)
    clip = np.full(250, 16384, np.int16)  # 0.5 in float
    task = asyncio.create_task(player.play(clip, 24000))
    await _wait_until(lambda: FakeStream.instances)
    stream = FakeStream.instances[0]
    first, second = stream.pull(200), stream.pull(200)
    await asyncio.wait_for(task, 1)
    assert np.allclose(first, 0.5) and np.allclose(second[:50], 0.5)
    assert np.allclose(second[50:], 0.0)  # tail of the clip, then silence
    assert stream.started and not stream.closed


async def test_player_reuses_one_stream_across_clips() -> None:
    import asyncio
    from jarvis.voice import audio

    FakeStream.instances = []
    player = audio.Player(stream_factory=FakeStream)
    for _ in range(3):
        task = asyncio.create_task(player.play(np.zeros(10, np.float32), 24000))
        await _wait_until(lambda: not player._drained.is_set())  # clip queued
        FakeStream.instances[0].pull(64)
        await asyncio.wait_for(task, 1)
    assert len(FakeStream.instances) == 1
    player.close()
    assert FakeStream.instances[0].closed


async def test_player_cancel_stops_audio() -> None:
    import asyncio
    from jarvis.voice import audio

    FakeStream.instances = []
    player = audio.Player(stream_factory=FakeStream)
    task = asyncio.create_task(player.play(np.ones(24000, np.float32), 24000))
    await _wait_until(lambda: FakeStream.instances)
    FakeStream.instances[0].pull(100)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert np.allclose(FakeStream.instances[0].pull(100), 0.0)  # nothing left queued
