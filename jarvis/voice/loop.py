"""The voice loop: IDLE -> LISTEN -> STT -> THINK/SPEAK -> FOLLOWUP -> IDLE.

Every dependency is injected so tests run on fakes. One shared frames iterator
serves the whole run (the mic queue is single-consumer): IDLE pulls frames for
wake detection, and the same iterator is handed to the recorder for LISTEN,
FOLLOWUP and spoken confirmations. All playback happens under ``mic.mute(0.3)``
so the assistant never hears (or wakes on) its own voice.
"""
from __future__ import annotations

import asyncio
import contextlib
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
# Said on its own, any of these drops the conversation without asking Claude.
CANCEL_PHRASES = {"nevermind", "never mind", "cancel", "stop", "forget it", "thats all"}

# What the conversation does after an utterance.
FOLLOWUP = "followup"  # answered: open the follow-up window
WAKE_ONLY = "wake"  # only the wake phrase was heard
IDLE = "idle"  # nothing to do, or the turn failed


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
        self._turn_active = False

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
            except Exception as exc:
                await self._fail(exc)

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
        in_followup = False
        while audio is not None:
            outcome = await self._process(audio)
            if outcome == WAKE_ONLY and in_followup:
                # "Hey Jarvis" alone in the follow-up window is a fresh wake.
                await self._tone("wake")
                self._set("LISTEN")
                audio = await self._record(LISTEN_TIMEOUT_S)
                in_followup = False
                continue
            if outcome != FOLLOWUP:
                return
            self._set("FOLLOWUP")
            audio = await self._record(self.cfg.followup_s)
            in_followup = True

    async def handle_turn(self, audio: np.ndarray) -> None:
        """Transcribe one utterance and answer it (exposed for tests)."""
        await self._process(audio)

    async def _process(self, audio: np.ndarray) -> str:
        """Handle one utterance; return FOLLOWUP, WAKE_ONLY or IDLE (what next)."""
        self._set("STT")
        try:
            transcript = await asyncio.to_thread(self.stt.transcribe, audio)
        except Exception as exc:
            await self._fail(exc)
            return IDLE
        self._archive(audio, transcript)
        text = strip_wake_phrase(transcript) if transcript else ""
        if not text:
            return WAKE_ONLY if transcript else IDLE
        if _normalize(text) in CANCEL_PHRASES:
            await self._say("Okay.")
            self._log_turn(text, "Okay.")
            return IDLE

        self._said = []
        outcome = FOLLOWUP
        self._turn_active = True
        try:
            self._set("THINK")
            # Timer starts before maybe_reset so a slow reconnect counts too.
            self._working = asyncio.create_task(self._working_tone())
            await self.brain.maybe_reset()
            if _normalize(text) == NEW_CONVERSATION:
                await self.brain.reset()
                await self._say("Okay, fresh start.")
            else:
                await self._answer(text)
        except Exception as exc:
            outcome = IDLE
            await self._fail(exc)
        finally:
            self._turn_active = False
            await self._stop_working_tone()
        self._log_turn(text, " ".join(self._said))
        return outcome

    def _log_turn(self, heard: str, said: str) -> None:
        try:
            self.log.log_turn(heard, said)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def _archive(self, audio: np.ndarray, transcript: str | None) -> None:
        """Archive an utterance; a disk failure is logged, never fails the turn."""
        try:
            self.log.archive_utterance(audio, transcript)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    async def _answer(self, text: str) -> None:
        splitter = SentenceSplitter()
        # aclosing: if speaking fails mid-stream, the brain's generator is closed
        # now (recycling its client), not whenever it is garbage collected.
        async with contextlib.aclosing(self.brain.ask(text)) as reply:
            async for delta in reply:
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

    async def confirm(self, question: str) -> bool | None:
        """Ask a yes/no question aloud. Only a clean affirmative proceeds.

        Returns None when nobody was asked (no turn is active), so the brain
        does not report it as a spoken decline.

        The SDK runs its permission callback in a detached task, so this can
        start while the turn is still speaking an earlier sentence: the audio
        lock makes it wait for that sentence, and holds speech back until the
        question has been asked and answered. If the brain cancels it (turn
        timeout), playback is stopped and the cancellation re-raised.
        """
        if not self._turn_active:
            # A stray callback outside a turn must never drive the shared
            # frames iterator alongside the IDLE wake detector.
            return None
        await self._stop_working_tone()
        try:
            async with self._audio_lock:
                await self._say_unlocked(question)
                audio = await self._record_unlocked(CONFIRM_TIMEOUT_S)
                transcript = None
                if audio is not None:
                    transcript = await asyncio.to_thread(self.stt.transcribe, audio)
                    self._archive(audio, transcript)
                answer = strip_wake_phrase(transcript) if transcript else None
                approved = is_affirmative(answer)
                if not approved:
                    await self._say_unlocked("Cancelled.")
                return approved
        except asyncio.CancelledError:
            self.player.stop()
            raise

    async def aclose(self) -> None:
        """Close the shared mic frames iterator so the input stream stops."""
        frames, self._frames = self._frames, None
        if frames is not None:
            await frames.aclose()  # type: ignore[attr-defined]

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
        self.recorder.reset()  # VAD state must not leak across utterances
        return await self.recorder.record(self._stream(), start_timeout_s)

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
