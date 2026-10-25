"""Tests for the end-of-utterance recorder and smart-turn model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jarvis.voice.turn import FRAME, SmartTurn, UtteranceRecorder

MODEL = Path(__file__).resolve().parents[2] / "models/smart_turn/smart-turn-v3.2-cpu.onnx"


class FakeVAD:
    """Speech flag per frame is its first sample (1 = speech)."""

    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(frame[0])


class FakeTurn:
    def __init__(self, probs: list[float]) -> None:
        self.probs = list(probs)
        self.calls = 0

    def predict(self, audio: np.ndarray) -> float:
        self.calls += 1
        return self.probs.pop(0) if len(self.probs) > 1 else self.probs[0]


def make_frames(spec: list[tuple[int, float]]):
    """spec: (is_speech, seconds) segments -> async iterator of frames."""

    async def gen():
        for flag, seconds in spec:
            for _ in range(round(seconds / (FRAME / 16000))):
                yield np.full(FRAME, flag, dtype=np.int16)

    return gen()


async def test_no_speech_returns_none():
    rec = UtteranceRecorder(FakeVAD(), FakeTurn([0.9]))
    assert await rec.record(make_frames([(0, 5.0)]), start_timeout_s=1.0) is None


async def test_pause_with_incomplete_turn_keeps_listening():
    turn = FakeTurn([0.1, 0.9])
    rec = UtteranceRecorder(FakeVAD(), turn, hard_silence_s=3.0, pause_check_s=0.4)
    audio = await rec.record(
        make_frames([(1, 1.0), (0, 0.5), (1, 1.0), (0, 1.0)]), start_timeout_s=2.0
    )
    assert turn.calls == 2
    # both speech spans (2.0 s) + the 0.5 s pause + 0.4 s of final silence = ~2.9 s
    assert audio is not None
    assert round(2.7 * 16000) <= len(audio) <= round(3.1 * 16000)


async def test_hard_silence_ends_turn():
    turn = FakeTurn([0.1])
    rec = UtteranceRecorder(FakeVAD(), turn, hard_silence_s=3.0, pause_check_s=0.4)
    audio = await rec.record(
        make_frames([(1, 1.0), (0, 10.0)]), start_timeout_s=2.0
    )
    assert audio is not None
    assert len(audio) == round(4.0 * 16000)
    assert turn.calls == 1  # one check per pause, not per frame


async def test_short_silence_never_consults_model():
    turn = FakeTurn([0.9])
    rec = UtteranceRecorder(FakeVAD(), turn, hard_silence_s=3.0, pause_check_s=0.4)
    audio = await rec.record(
        make_frames([(1, 1.0), (0, 0.24), (1, 0.5)]), start_timeout_s=2.0
    )
    assert turn.calls == 0
    assert audio is not None


async def test_continuous_speech_is_capped():
    # TV or music never pauses; the recording must still end.
    turn = FakeTurn([0.0])
    rec = UtteranceRecorder(FakeVAD(), turn, max_utterance_s=2.0)
    audio = await rec.record(make_frames([(1, 60.0)]), start_timeout_s=2.0)
    assert audio is not None
    assert round(1.9 * 16000) <= len(audio) <= round(2.1 * 16000)


async def test_max_utterance_defaults_to_thirty_seconds():
    rec = UtteranceRecorder(FakeVAD(), FakeTurn([0.0]))
    audio = await rec.record(make_frames([(1, 60.0)]), start_timeout_s=2.0)
    assert audio is not None
    assert round(29.9 * 16000) <= len(audio) <= round(30.1 * 16000)


class ResettableVAD(FakeVAD):
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def test_reset_resets_the_vad():
    vad = ResettableVAD()
    UtteranceRecorder(vad, FakeTurn([0.9])).reset()
    assert vad.resets == 1
    UtteranceRecorder(FakeVAD(), FakeTurn([0.9])).reset()  # no reset(): no-op


@pytest.mark.skipif(not MODEL.exists(), reason="smart-turn model not downloaded")
def test_smartturn_real_model_smoke():
    prob = SmartTurn(MODEL).predict(np.zeros(16000, dtype=np.int16))
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
