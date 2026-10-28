"""Stop-phrase barge-in tests with fake VAD, STT and frames."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from jarvis.voice.bargein import BargeIn, BargeInListener, find_stop_phrase

PHRASES = ["wait", "no", "hold up", "never mind", "jarvis"]
FRAME = 1280


def test_find_stop_phrase_whole_words() -> None:
    assert find_stop_phrase("Wait, stop", "", PHRASES) == "wait"
    assert find_stop_phrase("hold up a second", "", PHRASES) == "hold up"
    assert find_stop_phrase("Jarvis, stop", "", PHRASES) == "jarvis"
    assert find_stop_phrase("now I know", "", PHRASES) is None
    assert find_stop_phrase("hold the line up", "", PHRASES) is None
    assert find_stop_phrase("", "", PHRASES) is None


def test_find_stop_phrase_ignores_echo() -> None:
    assert find_stop_phrase("wait for it", "you should wait for it", PHRASES) is None
    assert find_stop_phrase("wait no", "you should wait for it", PHRASES) == "no"


class FakeVAD:
    def __init__(self, pattern):
        self.pattern, self.i, self.resets = pattern, 0, 0

    def reset(self):
        self.resets += 1

    def is_speech(self, frame):
        v = self.pattern[min(self.i, len(self.pattern) - 1)]
        self.i += 1
        return v


class FakeSTT:
    def __init__(self, text):
        self.text, self.calls = text, []

    def transcribe(self, audio):
        self.calls.append(len(audio))
        return self.text


class Frames:
    """Async iterator that records whether it was closed."""

    def __init__(self, n):
        self.n, self.i, self.closed = n, 0, False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.i >= self.n:
            raise StopAsyncIteration
        self.i += 1
        return np.full(FRAME, self.i, np.int16)

    async def aclose(self):
        self.closed = True


def _listener(vad, stt):
    return BargeInListener(vad, stt, PHRASES, min_speech_s=0.3, window_s=2.0, step_s=0.5)


async def test_stop_word_fires() -> None:
    stt = FakeSTT("wait a minute")
    frames = Frames(100)
    result = await _listener(FakeVAD([True]), stt).listen(frames, lambda: "hello there", asyncio.Event())
    assert isinstance(result, BargeIn) and result.phrase == "wait"
    assert result.audio.dtype == np.int16 and len(result.audio) >= int(0.3 * 16000)
    assert result.audio[0] == 1  # starts at the first speech frame
    assert not frames.closed


async def test_echo_is_ignored() -> None:
    stt = FakeSTT("wait for it")
    result = await _listener(FakeVAD([True]), stt).listen(Frames(40), lambda: "wait for it", asyncio.Event())
    assert result is None and stt.calls


async def test_short_blip_never_calls_stt() -> None:
    stt = FakeSTT("wait")
    vad = FakeVAD(([True, True] + [False] * 8) * 6)
    result = await _listener(vad, stt).listen(Frames(60), lambda: "", asyncio.Event())
    assert result is None and stt.calls == []


async def test_done_ends_without_closing() -> None:
    frames, done = Frames(100), asyncio.Event()
    done.set()
    result = await _listener(FakeVAD([False]), FakeSTT("wait")).listen(frames, lambda: "", done)
    assert result is None and not frames.closed and frames.i == 0
    done.clear()
    assert await frames.__anext__() is not None  # still usable


async def test_iterator_end_returns_none_and_resets_vad() -> None:
    vad = FakeVAD([False])
    result = await _listener(vad, FakeSTT("wait")).listen(Frames(3), lambda: "", asyncio.Event())
    assert result is None and vad.resets == 1


async def test_short_stop_word_is_transcribed_at_segment_end() -> None:
    stt = FakeSTT("no")
    vad = FakeVAD([True] * 5 + [False])
    result = await _listener(vad, stt).listen(Frames(40), lambda: "", asyncio.Event())
    assert result is not None and result.phrase == "no" and len(stt.calls) == 1


async def test_done_set_during_stt_returns_none() -> None:
    done = asyncio.Event()

    class SetsDone(FakeSTT):
        def transcribe(self, audio):
            done.set()
            return super().transcribe(audio)

    result = await _listener(FakeVAD([True]), SetsDone("wait")).listen(Frames(40), lambda: "", done)
    assert result is None


async def test_stt_cadence_over_long_segment() -> None:
    stt = FakeSTT("nothing")
    await _listener(FakeVAD([True]), stt).listen(Frames(50), lambda: "", asyncio.Event())
    assert 5 <= len(stt.calls) <= 8  # one per ~0.5 s once past min_speech_s
    assert max(stt.calls) <= 2 * 16000  # only the last window_s


async def test_stop_word_after_segment_reset_still_fires() -> None:
    class Seq(FakeSTT):
        def transcribe(self, audio):
            self.calls.append(len(audio))
            return "wait" if len(self.calls) >= 2 else "hello"

    vad = FakeVAD([True] * 5 + [False] * 8 + [True] * 5 + [False])
    stt = Seq("")
    result = await _listener(vad, stt).listen(Frames(60), lambda: "", asyncio.Event())
    assert result is not None and result.phrase == "wait" and result.audio[0] == 14


async def test_hit_audio_is_only_the_window_that_heard_it() -> None:
    # The segment began with Jarvis's own voice: keep just the STT window, not all of it.
    class Late(FakeSTT):
        def transcribe(self, audio):
            self.calls.append(len(audio))
            return "wait" if len(self.calls) >= 6 else "the weather today"

    stt = Late("")
    result = await _listener(FakeVAD([True]), stt).listen(Frames(100), lambda: "", asyncio.Event())
    assert result is not None and result.phrase == "wait"
    assert len(result.audio) == stt.calls[-1] <= 2 * 16000
    assert result.audio[0] > 1  # the first speech frames fell out of the window
