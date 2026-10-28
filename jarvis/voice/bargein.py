"""Stop-phrase barge-in: hear "wait" / "no" while Jarvis is speaking."""
from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from jarvis.voice.turn import FRAME, SAMPLE_RATE

SILENCE_END_S = 0.4  # this much non-speech ends a segment


class BargeIn(NamedTuple):
    phrase: str
    audio: np.ndarray  # int16 16 kHz speech captured since the speech began


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _contains(words: str, phrase: str) -> bool:
    return f" {phrase} " in f" {words} "


def find_stop_phrase(transcript: str, speaking: str, phrases: list[str]) -> str | None:
    """First stop phrase heard as whole words and not just echo of `speaking`."""
    heard, said = _normalize(transcript), _normalize(speaking)
    for phrase in phrases:
        p = _normalize(phrase)
        if p and _contains(heard, p) and not _contains(said, p):
            return phrase
    return None


class BargeInListener:
    def __init__(self, vad: Any, stt: Any, phrases: list[str], min_speech_s: float = 0.3,
                 window_s: float = 2.0, step_s: float = 0.5) -> None:
        self._vad, self._stt, self._phrases = vad, stt, phrases
        self._min = max(1, round(min_speech_s * SAMPLE_RATE / FRAME))
        self._window = max(1, round(window_s * SAMPLE_RATE / FRAME))
        self._step = max(1, round(step_s * SAMPLE_RATE / FRAME))
        self._silence_end = max(1, round(SILENCE_END_S * SAMPLE_RATE / FRAME))

    async def listen(self, frames: Any, speaking: Callable[[], str],
                     done: asyncio.Event) -> BargeIn | None:
        """Pull frames until a stop phrase is heard, `done` is set or the mic ends.

        `frames` is the shared mic iterator: it is only advanced, never closed.
        """
        if hasattr(self._vad, "reset"):
            self._vad.reset()
        segment: list[np.ndarray] = []
        speech = silence = since_stt = 0
        while not done.is_set():
            try:
                frame = await anext(frames)
            except StopAsyncIteration:
                return None
            if done.is_set():
                return None
            if self._vad.is_speech(frame):
                segment.append(frame)
                speech += 1
                silence = 0
            elif segment:
                segment.append(frame)
                silence += 1
                if silence >= self._silence_end:
                    segment, speech, silence, since_stt = [], 0, 0, 0
                continue
            else:
                continue
            since_stt += 1
            if speech < self._min or since_stt < self._step:
                continue
            since_stt = 0
            window = np.concatenate(segment[-self._window:])
            text = await asyncio.to_thread(self._stt.transcribe, window)
            phrase = find_stop_phrase(text or "", speaking(), self._phrases)
            if phrase:
                return BargeIn(phrase, np.concatenate(segment))
        return None
