"""End-of-utterance detection: Silero VAD, smart-turn model, hard silence stop."""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np

from jarvis.voice.trace import trace

SAMPLE_RATE = 16000
FRAME = 1280
SMART_TURN_SECONDS = 8
SMART_TURN_THRESHOLD = 0.5
# A short pause needs a confident "done" (mid-sentence "uh, like-" pauses score
# near the middle); the bar falls linearly to SMART_TURN_THRESHOLD as the pause grows.
STRICT_THRESHOLD = 0.8
STRICT_UNTIL_S = 0.24
SETTLED_AT_S = 1.0
RECHECK_S = 0.3  # re-ask the model this often while the pause continues
PREROLL_FRAMES = 3
MAX_UTTERANCE_S = 30.0


def end_threshold(silence_s: float) -> float:
    """Smart-turn probability needed to end the turn after ``silence_s`` of pause."""
    if silence_s <= STRICT_UNTIL_S:
        return STRICT_THRESHOLD
    if silence_s >= SETTLED_AT_S:
        return SMART_TURN_THRESHOLD
    frac = (silence_s - STRICT_UNTIL_S) / (SETTLED_AT_S - STRICT_UNTIL_S)
    return STRICT_THRESHOLD - frac * (STRICT_THRESHOLD - SMART_TURN_THRESHOLD)


class TurnEnd(NamedTuple):
    """Why the last recording stopped: smart_turn, hard_silence or max_length."""

    reason: str
    silence_s: float
    prob: float | None


class SpeechDetector(Protocol):
    def is_speech(self, frame: np.ndarray) -> bool: ...


class TurnModel(Protocol):
    def predict(self, audio: np.ndarray) -> float: ...


class SileroVAD:
    """Frame-level speech detector using openWakeWord's bundled Silero ONNX."""

    def __init__(self, threshold: float = 0.5) -> None:
        from openwakeword.vad import VAD

        self._vad = VAD()
        self._threshold = threshold

    def is_speech(self, frame: np.ndarray) -> bool:
        # 1280 samples is not a multiple of the 480 default; 640 splits it evenly.
        return float(self._vad.predict(frame, frame_size=640)) >= self._threshold

    def reset(self) -> None:
        self._vad.reset_states()


class SmartTurn:
    """Smart-turn v3 ONNX: probability that the speaker has finished their turn."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        self._extractor = WhisperFeatureExtractor(chunk_length=SMART_TURN_SECONDS)
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )

    def predict(self, audio: np.ndarray) -> float:
        """Completion probability in [0, 1] for int16 (or float) 16 kHz audio."""
        samples = SMART_TURN_SECONDS * SAMPLE_RATE
        audio = audio[-samples:]
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        features = self._extractor(
            audio.astype(np.float32),
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=samples,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        # The model's "logits" output already has a sigmoid applied (verified in [0, 1]).
        out = self._session.run(None, {"input_features": features})[0]
        return float(out[0][0])


class UtteranceRecorder:
    """Collects one utterance from a frame stream and decides when it is over."""

    def __init__(
        self,
        vad: SpeechDetector,
        smart_turn: TurnModel,
        hard_silence_s: float = 3.0,
        pause_check_s: float = 0.24,
        max_utterance_s: float = MAX_UTTERANCE_S,
    ) -> None:
        self._vad = vad
        self._turn = smart_turn
        self._hard_silence_s = hard_silence_s
        self._pause_check_s = pause_check_s
        self._max_utterance_s = max_utterance_s
        self.last_end: TurnEnd | None = None

    def reset(self) -> None:
        """Clear the VAD's recurrent state so it does not leak across utterances."""
        reset = getattr(self._vad, "reset", None)
        if callable(reset):
            reset()

    async def record(
        self, frames: AsyncIterator[np.ndarray], start_timeout_s: float
    ) -> np.ndarray | None:
        """Return the utterance audio, or None if speech never starts in time.

        Recording stops at the end of the turn, after hard silence, or once the
        utterance reaches ``max_utterance_s`` (continuous TV or music speech
        would otherwise never end it); the audio so far is returned.
        """
        frame_s = FRAME / SAMPLE_RATE
        timeout_frames = round(start_timeout_s / frame_s)
        max_frames = round(self._max_utterance_s / frame_s)
        pause_frames = round(self._pause_check_s / frame_s)
        hard_frames = round(self._hard_silence_s / frame_s)
        recheck_frames = max(1, round(RECHECK_S / frame_s))

        preroll: deque[np.ndarray] = deque(maxlen=PREROLL_FRAMES)
        chunks: list[np.ndarray] = []
        started = False
        waited = 0
        silent = 0
        next_check = pause_frames
        prob: float | None = None
        self.last_end = None

        async for frame in frames:
            speech = self._vad.is_speech(frame)
            if not started:
                if not speech:
                    preroll.append(frame)
                    waited += 1
                    if waited >= timeout_frames:
                        return None
                    continue
                started = True
                chunks.extend(preroll)
                chunks.append(frame)
                continue

            chunks.append(frame)
            if len(chunks) >= max_frames:
                self.last_end = TurnEnd("max_length", silent * frame_s, prob)
                break
            if speech:
                silent = 0
                next_check = pause_frames
                prob = None
                continue
            silent += 1
            if silent >= hard_frames:
                self.last_end = TurnEnd("hard_silence", silent * frame_s, prob)
                break
            if silent >= next_check:
                next_check = silent + recheck_frames
                audio = np.concatenate(chunks)
                prob = await asyncio.to_thread(self._turn.predict, audio)
                if prob >= end_threshold(silent * frame_s):
                    self.last_end = TurnEnd("smart_turn", silent * frame_s, prob)
                    break

        if self.last_end is not None:
            end = self.last_end
            score = "" if end.prob is None else f", smart-turn {end.prob:.2f}"
            trace("turn", f"{end.reason} after {end.silence_s:.2f} s silence{score}")
        return np.concatenate(chunks) if started else None
