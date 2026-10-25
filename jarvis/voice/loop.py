"""The voice loop: IDLE -> LISTEN -> STT -> THINK/SPEAK -> FOLLOWUP -> IDLE.

Every dependency is injected so tests run on fakes. One shared frames iterator
serves the whole run (the mic queue is single-consumer): IDLE pulls frames for
wake detection, and the same iterator is handed to the recorder for LISTEN,
FOLLOWUP and spoken confirmations. All playback happens under ``mic.mute(0.3)``
so the assistant never hears (or wakes on) its own voice.
"""
from __future__ import annotations

import asyncio
import re
import sys
import traceback
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from jarvis.voice.audio import tone
from jarvis.voice.brain import BrainTimeout
from jarvis.voice.permissions import is_affirmative
from jarvis.voice.speech import SentenceSplitter, to_speakable
from jarvis.voice.stt import strip_wake_phrase

LISTEN_TIMEOUT_S = 6
CONFIRM_TIMEOUT_S = 6
MUTE_TAIL_S = 0.3
NEW_CONVERSATION = "new conversation"


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


class VoiceLoop:
    def __init__(self, mic: Any, player: Any, wake: Any, recorder: Any, stt: Any,
                 brain: Any, tts: Any, log: Any, cfg: Any) -> None:
        self.mic = mic
        self.player = player
        self.wake = wake
        self.recorder = recorder
        self.stt = stt
        self.brain = brain
        self.tts = tts
        self.log = log
        self.cfg = cfg
        self.state = "IDLE"
        self._frames: AsyncIterator[np.ndarray] | None = None
        self._audio_lock = asyncio.Lock()
        self._working: asyncio.Task[None] | None = None
        self._said: list[str] = []

    # -- main loop ---------------------------------------------------------

    async def run(self) -> None:
        """Run until the mic stream ends. A failing turn never kills the loop."""
        frames = self._stream()
        while True:
            self._set("IDLE")
            self.wake.reset()
            if not await self._wait_for_wake(frames):
                return
            try:
                await self._conversation()
            except Exception:
                traceback.print_exc(file=sys.stderr)

    async def _wait_for_wake(self, frames: AsyncIterator[np.ndarray]) -> bool:
        async for frame in frames:
            if self.wake.process(frame):
                return True
        return False

    async def _conversation(self) -> None:
        """One wake: a command, then follow-ups until silence or an error."""
        self.wake.reset()
        await self._tone("wake")
        self._set("LISTEN")
        audio = await self._record(LISTEN_TIMEOUT_S)
        while audio is not None:
            if not await self._process(audio):
                return
            self._set("FOLLOWUP")
            audio = await self._record(self.cfg.followup_s)

    async def handle_turn(self, audio: np.ndarray) -> None:
        """Transcribe one utterance and answer it (exposed for tests)."""
        await self._process(audio)

    async def _process(self, audio: np.ndarray) -> bool:
        """Handle one utterance. True when the follow-up window should open."""
        self._set("STT")
        try:
            transcript = await asyncio.to_thread(self.stt.transcribe, audio)
            self.log.archive_utterance(audio, transcript)
        except Exception as exc:
            await self._fail(exc)
            return False
        text = strip_wake_phrase(transcript) if transcript else ""
        if not text:
            return False

        self._said = []
        followup = True
        try:
            self._set("THINK")
            await self.brain.maybe_reset()
            if _normalize(text) == NEW_CONVERSATION:
                await self.brain.reset()
                await self._say("Okay, fresh start.")
            else:
                await self._answer(text)
        except Exception as exc:
            followup = False
            await self._fail(exc)
        finally:
            await self._stop_working_tone()
        try:
            self.log.log_turn(text, " ".join(self._said))
        except Exception:
            traceback.print_exc(file=sys.stderr)
        return followup

    async def _answer(self, text: str) -> None:
        splitter = SentenceSplitter()
        self._working = asyncio.create_task(self._working_tone())
        async for delta in self.brain.ask(text):
            for sentence in splitter.feed(delta):
                await self._say(sentence)
        for sentence in splitter.flush():
            await self._say(sentence)

    async def _fail(self, exc: Exception) -> None:
        if isinstance(exc, BrainTimeout):
            message = "Sorry, that took too long."
        else:
            traceback.print_exception(exc, file=sys.stderr)
            message = f"Sorry, something broke: {type(exc).__name__}."
        await self._stop_working_tone()
        try:
            await self._tone("error")
            await self._say(message)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    # -- confirmation (called by the brain's permission callback) ----------

    async def confirm(self, question: str) -> bool:
        """Ask a yes/no question aloud. Only a clean affirmative proceeds.

        The SDK runs its permission callback in a detached task, so this can
        start while the turn is still speaking an earlier sentence: the audio
        lock makes it wait for that sentence, and holds speech back until the
        question has been asked and answered. If the brain cancels it (turn
        timeout), playback is stopped and the cancellation re-raised.
        """
        await self._stop_working_tone()
        try:
            async with self._audio_lock:
                await self._say_unlocked(question)
                audio = await self._record_unlocked(CONFIRM_TIMEOUT_S)
                transcript = None
                if audio is not None:
                    transcript = await asyncio.to_thread(self.stt.transcribe, audio)
                    self.log.archive_utterance(audio, transcript)
                answer = strip_wake_phrase(transcript) if transcript else None
                approved = is_affirmative(answer)
                if not approved:
                    await self._say_unlocked("Cancelled.")
                return approved
        except asyncio.CancelledError:
            self.player.stop()
            raise

    # -- audio helpers -----------------------------------------------------
    # One lock serializes everything that touches the speaker or the shared
    # frames iterator: speaking, tones, recording, and a whole confirmation.

    def _stream(self) -> AsyncIterator[np.ndarray]:
        if self._frames is None:
            self._frames = self.mic.frames()
        return self._frames

    async def _record(self, start_timeout_s: float) -> np.ndarray | None:
        async with self._audio_lock:
            return await self._record_unlocked(start_timeout_s)

    async def _record_unlocked(self, start_timeout_s: float) -> np.ndarray | None:
        self._reset_vad()
        return await self.recorder.record(self._stream(), start_timeout_s)

    def _reset_vad(self) -> None:
        # UtteranceRecorder exposes no reset of its own; its VAD (SileroVAD)
        # carries recurrent state that must not leak across utterances.
        vad = getattr(self.recorder, "_vad", None)
        reset = getattr(vad, "reset", None)
        if callable(reset):
            reset()

    async def _say(self, text: str) -> None:
        await self._stop_working_tone()
        async with self._audio_lock:
            await self._say_unlocked(text)

    async def _say_unlocked(self, text: str) -> None:
        speakable = to_speakable(text)
        if not speakable:
            return
        self._set("SPEAK")
        samples, sr = await asyncio.to_thread(self.tts.synth, speakable)
        with self.mic.mute(MUTE_TAIL_S):
            await self.player.play(samples, sr)
        self._said.append(speakable)

    async def _tone(self, kind: str) -> None:
        samples, sr = tone(kind)  # type: ignore[arg-type]
        async with self._audio_lock:
            with self.mic.mute(MUTE_TAIL_S):
                await self.player.play(samples, sr)

    async def _working_tone(self) -> None:
        await asyncio.sleep(self.cfg.working_tone_after_s)
        await self._tone("working")

    async def _stop_working_tone(self) -> None:
        task, self._working = self._working, None
        if task is None or task is asyncio.current_task():
            return
        task.cancel()
        # gather (not a bare await) so the tone's cancellation or failure is
        # absorbed while a cancellation of this coroutine still propagates.
        await asyncio.gather(task, return_exceptions=True)

    def _set(self, state: str) -> None:
        if state != self.state:
            print(f"[voice] {self.state} -> {state}", file=sys.stderr, flush=True)
            self.state = state
