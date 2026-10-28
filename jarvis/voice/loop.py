"""The voice loop: IDLE -> LISTEN -> STT -> THINK/SPEAK -> FOLLOWUP -> IDLE.

Every dependency is injected so tests run on fakes. One shared frames iterator
serves the whole run (the mic queue is single-consumer): IDLE pulls frames for
wake detection, and the same iterator is handed to the recorder for LISTEN,
FOLLOWUP and spoken confirmations. Playback happens under ``mic.mute(0.3)``
so the assistant never hears (or wakes on) its own voice, except reply sentences
when a barge-in listener is set: those play unmuted while it listens on the same
iterator for a stop phrase ("wait", "hold up") that cuts the reply off.
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import sys
import time
import traceback
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from jarvis.voice.audio import tone
from jarvis.voice.brain import BrainTimeout
from jarvis.voice.config import DEFAULT_STOP_PHRASES
from jarvis.voice.intent import judge_approval
from jarvis.voice.trace import trace
from jarvis.voice.speech import SentenceSplitter, to_speakable
from jarvis.voice.stt import strip_wake_phrase

LISTEN_TIMEOUT_S = 6
CONFIRM_TIMEOUT_S = 6
# A transcript that trails off ("check out, uh, like-") is a pause, not the end:
# keep listening up to MAX_CONTINUATIONS times for the rest of the sentence.
CONTINUE_TIMEOUT_S = 3.0
MAX_CONTINUATIONS = 2
TRAILING_WORDS = {"uh", "um", "uhm", "umm", "er", "erm", "hmm", "like", "and", "but", "or",
                  "because", "the", "a", "an", "to", "of", "with", "for"}
MUTE_TAIL_S = 0.3
# The re-duck before a sentence may delay it by at most this past the synth: a hung
# music app's osascript keeps running in the background instead.
DUCK_WAIT_S = 0.3
# Mic audio queued longer than this since the last reply sentence (Claude thinking,
# a tool call) is dropped before listening for stop phrases again.
STALE_AUDIO_S = 1.0
NEW_CONVERSATION = "new conversation"
# Said on its own, any of these drops the conversation without asking Claude.
CANCEL_PHRASES = {"nevermind", "never mind", "cancel", "stop", "forget it", "thats all"}
# Short utterances containing one of these are a cancel too ("ah nevermind then").
CANCEL_WORDS = ("nevermind", "never mind", "forget it", "forget about it", "cancel that")
CANCEL_MAX_WORDS = 6
# Claude is told to answer a vaguer brush-off with just "Okay." and thanks with
# "You're welcome."; either reply ends the conversation too.
DISMISSED_REPLIES = {"okay", "okay sir", "ok", "ok sir", "youre welcome", "youre welcome sir"}
# An utterance made only of these (and filler) is a goodbye: "thanks", "sounds good".
CLOSING_PHRASES = (
    "thank you", "thanks", "thank u", "thx", "ty", "appreciate it", "sounds good",
    "sounds great", "got it", "cool", "perfect", "great", "awesome", "nice", "sweet",
    "bet", "alright", "all right", "okay", "ok", "cheers", "bye", "goodbye", "good night",
    "goodnight", "thats all", "thats it", "that is all", "im good", "all good", "were good",
    "love it", "nice one",
)
CLOSING_FILLER = {"jarvis", "claude", "hey", "sir", "man", "bro", "dude", "so", "much",
                  "very", "a", "lot", "then", "yeah", "yep", "and", "for", "that", "now"}
# Said around stop phrases ("uh, wait"), these still leave nothing for Claude.
STOP_FILLER = {"uh", "um", "umm", "oh", "ah", "hey", "sir"}
_CANCEL_WORD = re.compile(r"\b(" + "|".join(map(re.escape, CANCEL_WORDS)) + r")\b")
_CLOSING = re.compile(r"\b(" + "|".join(sorted(map(re.escape, CLOSING_PHRASES), key=len,
                                               reverse=True)) + r")\b")


def _is_cancel(text: str) -> bool:
    norm = _normalize(text)
    if norm in CANCEL_PHRASES:
        return True
    if len(norm.split()) > CANCEL_MAX_WORDS:
        return False
    ends = [m.end() for m in _CANCEL_WORD.finditer(norm)]
    # "never mind, play rock" is a new request, not a cancel.
    return bool(ends) and all(w in CLOSING_FILLER | STOP_FILLER for w in norm[max(ends):].split())


def _is_closing(text: str) -> bool:
    """Only thanks / sounds good / that's all, with filler: the conversation is over."""
    norm = _normalize(text)
    rest, found = _CLOSING.subn(" ", norm)
    return found > 0 and all(w in CLOSING_FILLER for w in rest.split())


def _only_stop_phrases(text: str, phrases: tuple[str, ...]) -> bool:
    words = f" {_normalize(text)} "
    for phrase in sorted(map(_normalize, phrases), key=len, reverse=True):
        if phrase:
            words = words.replace(f" {phrase} ", " ")
    return all(w in STOP_FILLER for w in words.split())


def _trails_off(text: str) -> bool:
    """True if the speaker stopped mid-thought: a dash or ellipsis, or a trailing
    filler with no full stop, question mark or exclamation mark after it."""
    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.endswith(("-", "\u2014", "...", "\u2026")):
        return True
    if stripped.endswith((".", "?", "!")):
        return False  # Whisper ended the sentence: "What's the weather like?" is finished
    words = _normalize(stripped).split()
    return bool(words) and words[-1] in TRAILING_WORDS



def _trim_echo(transcript: str, phrase: str, said: str) -> str:
    """Drop Jarvis's own words from the front of a post-interrupt transcript.

    The listener's VAD hears the speaker, so the captured audio (and the transcript
    of it plus the continuation) begins with the words played before the stop
    phrase: "The weather today is sunny and wait". Keep from the first whole-word
    `phrase` on; if it is not there, drop a leading run of the sentence's words.
    """
    tokens = transcript.split()
    norms = [_normalize(t) for t in tokens]
    target = _normalize(phrase).split()
    if target:
        for i in range(len(tokens) - len(target) + 1):
            if norms[i:i + len(target)] == target:
                return " ".join(tokens[i:])
    played = _normalize(said).split()
    i = j = 0
    while i < len(tokens):
        if norms[i]:
            if norms[i] not in played[j:]:
                break
            j = played.index(norms[i], j) + 1
        i += 1
    return " ".join(tokens[i:])


# What the conversation does after an utterance.
FOLLOWUP = "followup"  # answered: open the follow-up window
WAKE_ONLY = "wake"  # only the wake phrase was heard
IDLE = "idle"  # nothing to do, or the turn failed
INTERRUPTED = "interrupted"  # a stop phrase cut the reply off: record the rest


class Interrupted(Exception):
    """A stop phrase was heard during a reply sentence."""

    def __init__(self, audio: np.ndarray, phrase: str = "", sentence: str = "") -> None:
        super().__init__("interrupted")
        self.audio = audio  # the window that heard the phrase (often with Jarvis's echo)
        self.phrase = phrase  # the stop phrase that was heard
        self.sentence = sentence  # the sentence being played when it was heard


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


class VoiceLoop:
    def __init__(self, mic: Any, player: Any, wake: Any, recorder: Any, stt: Any,
                 brain: Any, tts: Any, log: Any, cfg: Any, judge: Any = None,
                 ducker: Any = None, barge: Any = None) -> None:
        self.mic = mic
        self.player = player
        self.wake = wake
        self.recorder = recorder
        self.stt = stt
        self.brain = brain
        self.tts = tts
        self.log = log
        self.cfg = cfg
        self.judge = judge or judge_approval  # (question, answer) -> approved?
        self.ducker = ducker  # lowers other apps' music while engaged; None = off
        self.barge = barge  # stop-phrase listener for reply sentences; None = always muted
        self.state = "IDLE"
        self._frames: AsyncIterator[np.ndarray] | None = None
        self._audio_lock = asyncio.Lock()
        self._working: asyncio.Task[None] | None = None
        self._ducking: asyncio.Task[None] | None = None  # the duck in flight, if any
        self._said: list[str] = []  # everything spoken this turn, confirmations too
        self._reply: list[str] = []  # just Claude's reply sentences, as played
        self._turn_active = False
        self._cut: str | None = None  # what Claude's reply got out before an interrupt
        self._interrupt: Interrupted | None = None
        self._cutting = False  # a reply was just cut off: queued confirmations are denied
        self._clock = time.monotonic
        self._listened_at: float | None = None  # when the last listened sentence ended

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
        # Duck alongside the wake tone so osascript never delays it; the finally
        # waits for the duck so it cannot land after the restore.
        self._start_duck()
        try:
            await self._tone("wake")
            await self._converse()
        finally:
            ducking = self._ducking
            if ducking is not None:
                await asyncio.gather(ducking, return_exceptions=True)
            await self._restore()

    async def _converse(self) -> None:
        self._cut, self._interrupt = None, None
        self._set("LISTEN")
        audio = await self._record(LISTEN_TIMEOUT_S)
        in_followup = False
        while audio is not None:
            outcome = await self._process(audio)
            if outcome == INTERRUPTED:
                # The user is still talking: record the rest after what was captured.
                captured = self._interrupt.audio  # type: ignore[union-attr]
                self._set("LISTEN")
                more = await self._record(self.cfg.followup_s)
                if more is None:
                    return
                audio = np.concatenate([captured, more])
                in_followup = True
                continue
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

    def _start_duck(self) -> asyncio.Task[None] | None:
        """Duck in the background; one osascript run at a time (a hung app must not
        queue a call per sentence behind it)."""
        if self.ducker is None:
            return None
        if self._ducking is None or self._ducking.done():
            self._ducking = asyncio.create_task(self._duck())
        return self._ducking

    async def _duck(self) -> None:
        if self.ducker is None:
            return
        try:
            await asyncio.to_thread(self.ducker.duck)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    async def _restore(self) -> None:
        if self.ducker is None:
            return
        try:
            await asyncio.to_thread(self.ducker.restore)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    async def handle_turn(self, audio: np.ndarray) -> None:
        """Transcribe one utterance and answer it (exposed for tests)."""
        await self._process(audio)

    async def _process(self, audio: np.ndarray) -> str:
        """Handle one utterance; return FOLLOWUP, WAKE_ONLY or IDLE (what next)."""
        self._set("STT")
        self._cutting = False
        try:
            transcript = await asyncio.to_thread(self.stt.transcribe, audio)
            for _ in range(MAX_CONTINUATIONS):
                if not transcript or not _trails_off(transcript):
                    break
                if getattr(getattr(self.recorder, "last_end", None), "reason", "") == "max_length":
                    break  # capped (TV, music): there is no pause to wait out
                trace("turn", "trailed off, still listening")
                self._set("LISTEN")
                more = await self._record(CONTINUE_TIMEOUT_S)
                if more is None:
                    break
                audio = np.concatenate([audio, more])
                self._set("STT")
                transcript = await asyncio.to_thread(self.stt.transcribe, audio)
        except Exception as exc:
            await self._fail(exc)
            return IDLE
        self._archive(audio, transcript)
        cut, self._interrupt = self._interrupt, None
        after_cut = cut is not None
        if cut is not None and transcript:
            # What the mic caught of Jarvis's own voice is not the user speaking.
            transcript = _trim_echo(transcript, cut.phrase, cut.sentence) or None
        trace("heard", transcript or "(nothing)")
        text = strip_wake_phrase(transcript) if transcript else ""
        if not text:
            if after_cut:
                return FOLLOWUP  # "Jarvis" alone: it stopped, now it listens
            return WAKE_ONLY if transcript else IDLE
        if _is_cancel(text):
            await self._say("Okay.")
            self._log_turn(text, "Okay.")
            return IDLE
        # A goodbye ends the follow-up, unless it answers a question we just asked
        # ("Want me to set a timer?" "Sounds good."): that goes to Claude.
        asked = " ".join(self._said).rstrip().endswith("?")
        if _is_closing(text) and not asked:
            reply = ""
            if "thank" in _normalize(text):
                sir = getattr(self.cfg, "address_as", "")
                reply = f"You're welcome, {sir}." if sir else "You're welcome."
                await self._say(reply)
            self._log_turn(text, reply)
            return IDLE
        if after_cut and _only_stop_phrases(
                text, getattr(self.cfg, "stop_phrases", DEFAULT_STOP_PHRASES)):
            return FOLLOWUP  # "wait" / "hold up": it stopped, now it listens

        self._said, self._reply = [], []
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
                await self._answer(self._cut_note() + text)
                if _normalize(" ".join(self._said)) in DISMISSED_REPLIES:
                    outcome = IDLE  # Claude read it as a brush-off: end the conversation
        except Interrupted as exc:
            self._interrupt = exc
            self._cut = " ".join(self._reply)
            outcome = INTERRUPTED
        except Exception as exc:
            outcome = IDLE
            await self._fail(exc)
        finally:
            self._turn_active = False
            await self._stop_working_tone()
        self._log_turn(text, " ".join(self._said))
        return outcome

    def _cut_note(self) -> str:
        """Tell Claude where its last reply was cut off (once, on the next turn)."""
        cut, self._cut = self._cut, None
        if cut is None:
            return ""
        if not cut:
            return "(You were interrupted before saying anything. The rest was not heard.) "
        return f'(You were interrupted after saying: "{cut}". The rest was not heard.) '

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
        self._listened_at = None
        # aclosing: if speaking fails mid-stream, the brain's generator is closed
        # now (recycling its client), not whenever it is garbage collected.
        async with contextlib.aclosing(self.brain.ask(text)) as reply:
            try:
                async for delta in reply:
                    for sentence in splitter.feed(delta):
                        await self._say(sentence, barge_in=True)
            except Interrupted:
                # Before the close: the brain then drains the reply, keeping its session.
                try:
                    await self.brain.interrupt()
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                raise
        for sentence in splitter.flush():
            await self._say(sentence, barge_in=True)

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
        """Ask a yes/no question aloud; the answer's intent decides (see intent.py).

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
                if self._cutting:
                    # Queued behind the sentence the user just cut off: they said
                    # "wait", so deny without asking and keep the mic for their next words.
                    return False
                await self._say_unlocked(question)
                audio = await self._record_unlocked(CONFIRM_TIMEOUT_S)
                transcript = None
                if audio is not None:
                    transcript = await asyncio.to_thread(self.stt.transcribe, audio)
                    self._archive(audio, transcript)
                trace("heard", transcript or "(nothing)")
                answer = strip_wake_phrase(transcript) if transcript else None
                approved = await self.judge(question, answer)
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

    async def _say(self, text: str, barge_in: bool = False) -> None:
        await self._stop_working_tone()
        async with self._audio_lock:
            await self._say_unlocked(text, barge_in)

    async def _say_unlocked(self, text: str, barge_in: bool = False) -> None:
        speakable = to_speakable(text)
        if not speakable:
            return
        self._set("SPEAK")
        # Re-duck while synthesizing: music Claude started mid-turn gets lowered too.
        ducking = self._start_duck()
        samples, sr = await asyncio.to_thread(self.tts.synth, speakable)
        if ducking is not None and not ducking.done():
            # asyncio.wait never cancels it: a slow duck finishes in the background.
            await asyncio.wait({ducking}, timeout=DUCK_WAIT_S)
        trace("said", speakable)
        if not barge_in or self.barge is None:
            with self.mic.mute(MUTE_TAIL_S):
                await self.player.play(samples, sr)
            self._said.append(speakable)
            if barge_in:
                self._reply.append(speakable)
            return
        if self._listened_at is None or self._clock() - self._listened_at > STALE_AUDIO_S:
            with self.mic.mute(MUTE_TAIL_S):
                pass  # drop frames queued while Claude was thinking: they are not a reply to this
        # Otherwise keep what queued since the last sentence: a "wait" said in between counts.
        hit, fraction = await self._play_listening(samples, sr, speakable)
        if hit is None:
            with self.mic.mute(MUTE_TAIL_S):
                pass  # tail only: the room's echo of the last word is not the user
            self._listened_at = self._clock()
            self._said.append(speakable)
            self._reply.append(speakable)
            return
        # No tail here: what the mic hears next is the user still talking.
        trace("interrupt", hit.phrase)
        words = speakable.split()
        played = " ".join(words[:round(len(words) * fraction)])
        if played:
            self._said.append(played)
            self._reply.append(played)
        self._cutting = True
        raise Interrupted(hit.audio, hit.phrase, speakable)

    async def _play_listening(self, samples: Any, sr: int, speakable: str) -> tuple[Any, float]:
        """Play a clip while the barge-in listener pulls the mic; stop on a stop phrase."""
        done = asyncio.Event()
        play = asyncio.ensure_future(self.player.play(samples, sr))
        listen = asyncio.ensure_future(self.barge.listen(self._stream(), lambda: speakable, done))
        hit = None
        try:
            await asyncio.wait({play, listen}, return_when=asyncio.FIRST_COMPLETED)
            if listen.done() and not listen.cancelled() and listen.exception() is None:
                hit = listen.result()
            if hit is not None:
                self.player.stop()
            fraction = await play
        finally:
            if not play.done():
                play.cancel()
            done.set()
            # Never cancel the listener: cancelling it mid-read would close the
            # shared frames iterator. Once `done` is set it returns on the next frame.
            await asyncio.gather(play, listen, return_exceptions=True)
            if listen.done() and not listen.cancelled() and listen.exception() is not None:
                traceback.print_exception(listen.exception(), file=sys.stderr)
        return hit, 1.0 if fraction is None else float(fraction)

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
