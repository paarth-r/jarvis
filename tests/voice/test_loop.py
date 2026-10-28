"""Tests for the voice-loop state machine, with every dependency faked."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.voice.audio import tone
from jarvis.voice.brain import BrainTimeout
from jarvis.voice.intent import judge_approval
from jarvis.voice.loop import VoiceLoop

TONES = {kind: tone(kind)[0] for kind in ("wake", "working", "error")}


class FakeMic:
    """Frames whose first sample is 1 trigger the fake wake word."""

    def __init__(self, wakes: int = 1) -> None:
        self.script = [np.full(1280, 1, dtype=np.int16)] * wakes
        self.muted = False
        self.mute_calls = 0

    async def frames(self):
        for frame in self.script:
            yield frame

    @contextmanager
    def mute(self, seconds_tail: float = 0.3):
        assert seconds_tail == 0.3
        self.mute_calls += 1
        self.muted = True
        try:
            yield
        finally:
            self.muted = False


class FakePlayer:
    def __init__(self, mic: FakeMic) -> None:
        self.mic = mic
        self.played: list[str] = []
        self.muted_flags: list[bool] = []

    async def play(self, samples, sr) -> None:
        self.muted_flags.append(self.mic.muted)
        if isinstance(samples, str):
            self.played.append(samples)
            return
        for kind, ref in TONES.items():
            if np.array_equal(samples, ref):
                self.played.append(f"tone:{kind}")
                return
        self.played.append("tone:?")

    def stop(self) -> None:
        pass


class FakeWake:
    def __init__(self) -> None:
        self.resets = 0

    def process(self, frame):
        return "hey_jarvis" if frame[0] == 1 else None

    def reset(self) -> None:
        self.resets += 1


class FakeRecorder:
    """Returns scripted utterances; None once the script runs out."""

    def __init__(self, utterances) -> None:
        self.utterances = list(utterances)
        self.timeouts: list[float] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    async def record(self, frames, start_timeout_s):
        self.timeouts.append(start_timeout_s)
        return self.utterances.pop(0) if self.utterances else None


class FakeSTT:
    def __init__(self, transcripts) -> None:
        self.transcripts = list(transcripts)

    def transcribe(self, audio):
        return self.transcripts.pop(0) if self.transcripts else None


class FakeTTS:
    def synth(self, text):
        return text, 24000


class FakeBrain:
    def __init__(self, replies=None, error: Exception | None = None, delay: float = 0.0):
        self.replies = list(replies or [])
        self.error = error
        self.delay = delay
        self.asked: list[str] = []
        self.resets = 0
        self.maybe_resets = 0
        self.confirm = None

    async def maybe_reset(self) -> None:
        self.maybe_resets += 1

    async def reset(self) -> None:
        self.resets += 1

    async def ask(self, text):
        self.asked.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        for delta in self.replies.pop(0) if self.replies else []:
            yield delta


class FakeLog:
    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []
        self.archived: list[str | None] = []

    def log_turn(self, heard, said) -> None:
        self.turns.append((heard, said))

    def archive_utterance(self, audio, transcript):
        self.archived.append(transcript)


CFG = SimpleNamespace(followup_s=8.0, working_tone_after_s=3.0, address_as="sir")
AUDIO = np.ones(1600, dtype=np.int16)


def fake_judge(verdict="DENY"):
    """judge_approval with the Claude call replaced by a fixed verdict."""
    calls: list[str] = []

    async def ask(question, answer):
        calls.append(answer)
        return verdict

    async def judge(question, answer):
        return await judge_approval(question, answer, ask=ask)

    judge.calls = calls
    return judge


def make(utterances, transcripts, brain, wakes=1, cfg=CFG, judge=None):
    mic = FakeMic(wakes)
    player = FakePlayer(mic)
    deps = SimpleNamespace(
        mic=mic, player=player, wake=FakeWake(), recorder=FakeRecorder(utterances),
        stt=FakeSTT(transcripts), brain=brain, tts=FakeTTS(), log=FakeLog(),
    )
    loop = VoiceLoop(deps.mic, deps.player, deps.wake, deps.recorder, deps.stt,
                     deps.brain, deps.tts, deps.log, cfg, judge=judge or fake_judge())
    return loop, deps


async def test_wake_to_answer():
    brain = FakeBrain([["Report is ", "due. By", "e."]])
    loop, d = make([AUDIO], ["Hey Jarvis, what's due?"], brain)
    await loop.run()
    assert brain.asked == ["what's due?"]
    assert d.player.played == ["tone:wake", "Report is due.", "Bye."]
    assert d.log.turns == [("what's due?", "Report is due. Bye.")]
    assert d.log.archived == ["Hey Jarvis, what's due?"]
    assert d.wake.resets >= 1
    assert d.recorder.timeouts == [6, 8.0]  # LISTEN, then the follow-up window
    assert d.recorder.resets == 2  # recorder reset before every record()
    assert brain.maybe_resets == 1


async def test_phantom_transcript_no_brain_call():
    brain = FakeBrain([["Hi."]])
    loop, d = make([AUDIO], [None], brain)
    await loop.run()
    assert brain.asked == []
    assert d.player.played == ["tone:wake"]
    assert d.log.archived == [None]  # archived even with no transcript
    assert d.log.turns == []
    assert d.recorder.timeouts == [6]  # straight back to IDLE, no follow-up


async def test_wake_phrase_only_is_no_command():
    brain = FakeBrain([["Hi."]])
    loop, d = make([AUDIO], ["Hey Jarvis."], brain)
    await loop.run()
    assert brain.asked == []


async def test_followup_without_wake():
    brain = FakeBrain([["Report is due."], ["Nothing tomorrow."]])
    loop, d = make([AUDIO, AUDIO], ["what's due", "and tomorrow"], brain)
    await loop.run()
    assert brain.asked == ["what's due", "and tomorrow"]
    assert d.player.played.count("tone:wake") == 1
    assert d.player.played[-1] == "Nothing tomorrow."
    assert d.recorder.timeouts == [6, 8.0, 8.0]


async def test_timeout_speaks_error_and_idles():
    brain = FakeBrain(error=BrainTimeout("slow"))
    loop, d = make([AUDIO, AUDIO], ["what's due", "second wake"], brain, wakes=2)
    await loop.run()
    assert d.player.played[:3] == ["tone:wake", "tone:error", "Sorry, that took too long."]
    # Back to IDLE (no follow-up record): the second command needed a new wake.
    assert d.recorder.timeouts == [6, 6]
    assert d.player.played.count("tone:wake") == 2
    assert brain.asked == ["what's due", "second wake"]


async def test_other_exception_names_type_and_loop_survives():
    brain = FakeBrain(error=RuntimeError("boom"))
    loop, d = make([AUDIO], ["what's due"], brain)
    await loop.run()
    assert d.player.played[1:] == ["tone:error", "Sorry, something broke: RuntimeError."]


async def test_mic_muted_while_speaking():
    brain = FakeBrain([["One. Two. Three."]])
    loop, d = make([AUDIO], ["count"], brain)
    await loop.run()
    assert len(d.player.played) == 4
    assert all(d.player.muted_flags)


async def test_working_tone_when_slow():
    cfg = SimpleNamespace(followup_s=8.0, working_tone_after_s=0.01)
    brain = FakeBrain([["Done."]], delay=0.1)
    loop, d = make([AUDIO], ["slow thing"], brain, cfg=cfg)
    await loop.run()
    assert d.player.played == ["tone:wake", "tone:working", "Done."]


async def test_no_working_tone_when_fast():
    brain = FakeBrain([["Done."]])
    loop, d = make([AUDIO], ["fast thing"], brain)
    await loop.run()
    assert "tone:working" not in d.player.played


async def test_confirm_declines_on_unclear():
    loop, d = make([AUDIO], ["yes no wait"], FakeBrain())
    loop._turn_active = True
    assert await loop.confirm("Send the email to Sam?") is False
    assert d.player.played == ["Send the email to Sam?", "Cancelled."]
    assert all(d.player.muted_flags)
    assert d.recorder.timeouts == [6]
    assert d.log.archived == ["yes no wait"]


async def test_confirm_accepts_affirmative():
    loop, d = make([AUDIO], ["Yes."], FakeBrain())
    loop._turn_active = True
    assert await loop.confirm("Send it?") is True
    assert d.player.played == ["Send it?"]


async def test_confirm_silence_cancels():
    loop, d = make([], [], FakeBrain())
    loop._turn_active = True
    assert await loop.confirm("Send it?") is False
    assert d.player.played == ["Send it?", "Cancelled."]


async def test_new_conversation_resets():
    brain = FakeBrain([["Hi."]])
    loop, d = make([AUDIO], ["New conversation."], brain)
    await loop.run()
    assert brain.resets == 1
    assert brain.asked == []
    assert d.player.played == ["tone:wake", "Okay, fresh start."]
    assert d.log.turns == [("New conversation.", "Okay, fresh start.")]


@pytest.mark.parametrize("phrase", ["Nevermind.", "never mind", "Hey Jarvis, cancel.", "Stop!"])
async def test_nevermind_drops_the_conversation(phrase):
    brain = FakeBrain([["Hi."]])
    loop, d = make([AUDIO], [phrase], brain)
    await loop.run()
    assert brain.asked == []
    assert d.player.played == ["tone:wake", "Okay."]


async def test_nevermind_in_followup_ends_it():
    brain = FakeBrain([["Report is due."], ["unused"]])
    loop, d = make([AUDIO, AUDIO], ["what's due", "nevermind"], brain)
    await loop.run()
    assert brain.asked == ["what's due"]
    assert d.player.played == ["tone:wake", "Report is due.", "Okay."]


async def test_handle_turn_directly():
    brain = FakeBrain([["Sure."]])
    loop, d = make([], ["do it"], brain)
    await loop.handle_turn(AUDIO)
    assert brain.asked == ["do it"]
    assert d.player.played == ["Sure."]


class ConfirmingBrain(FakeBrain):
    """Calls the loop's confirm from inside ask, like the SDK permission hook."""

    async def ask(self, text):
        self.asked.append(text)
        ok = await self.confirm("Send it?")
        yield "Sent." if ok else "Not sent."


async def test_confirm_from_inside_ask():
    brain = ConfirmingBrain()
    loop, d = make([AUDIO, AUDIO], ["email Sam", "yes"], brain)
    brain.confirm = loop.confirm
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played == ["tone:wake", "Send it?", "Sent."]
    assert d.log.archived == ["email Sam", "yes"]
    assert d.log.turns == [("email Sam", "Send it? Sent.")]


class EventPlayer(FakePlayer):
    """Logs start/end of every play; plays take a little real time."""

    def __init__(self, mic: FakeMic) -> None:
        super().__init__(mic)
        self.events: list[str] = []
        self.stops = 0

    async def play(self, samples, sr) -> None:
        name = samples if isinstance(samples, str) else "tone"
        self.events.append(f"start:{name}")
        await asyncio.sleep(0.02)
        await super().play(samples, sr)
        self.events.append(f"end:{name}")

    def stop(self) -> None:
        self.stops += 1


async def test_confirm_waits_for_sentence_in_progress():
    loop, d = make([AUDIO], ["yes"], FakeBrain())
    d.player = loop.player = EventPlayer(d.mic)
    speaking = asyncio.create_task(loop._say("First sentence."))
    await asyncio.sleep(0.005)  # the sentence is now mid-play
    loop._turn_active = True
    assert await loop.confirm("Send it?") is True
    await speaking
    assert d.player.events == ["start:First sentence.", "end:First sentence.",
                               "start:Send it?", "end:Send it?"]


class BlockingRecorder(FakeRecorder):
    """First record() hangs until cancelled; later calls follow the script."""

    def __init__(self, utterances) -> None:
        super().__init__(utterances)
        self.blocked = asyncio.Event()

    async def record(self, frames, start_timeout_s):
        if not self.blocked.is_set():
            self.blocked.set()
            await asyncio.Event().wait()
        return await super().record(frames, start_timeout_s)


async def test_cancelled_confirm_releases_lock():
    brain = FakeBrain([["Still here."]])
    loop, d = make([], ["next command"], brain)
    d.player = loop.player = EventPlayer(d.mic)
    recorder = BlockingRecorder([])
    loop.recorder = recorder
    loop._turn_active = True
    task = asyncio.create_task(loop.confirm("Send it?"))
    await recorder.blocked.wait()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("confirm swallowed its cancellation")
    assert d.player.stops == 1
    assert not loop._audio_lock.locked()
    await asyncio.wait_for(loop.handle_turn(AUDIO), timeout=1)
    assert d.player.played[-1] == "Still here."


async def test_wake_phrase_alone_in_followup_relistens():
    brain = FakeBrain([["Report is due."], ["Nothing tomorrow."]])
    loop, d = make([AUDIO, AUDIO, AUDIO], ["what's due", "Hey Jarvis.", "and tomorrow"], brain)
    await loop.run()
    assert d.player.played.count("tone:wake") == 2
    assert d.recorder.timeouts == [6, 8.0, 6, 8.0]
    assert brain.asked == ["what's due", "and tomorrow"]


async def test_archive_failure_does_not_fail_turn():
    brain = FakeBrain([["Sure."]])
    loop, d = make([], ["do it"], brain)

    def broken(audio, transcript):
        raise OSError("disk full")

    d.log.archive_utterance = broken
    await loop.handle_turn(AUDIO)
    assert brain.asked == ["do it"]
    assert d.player.played == ["Sure."]


async def test_confirm_outside_a_turn_asks_nobody():
    loop, d = make([AUDIO], ["yes"], FakeBrain())
    # None (not False): nobody was asked, so it must not read as a decline.
    assert await loop.confirm("Send it?") is None
    assert d.player.played == []
    assert d.recorder.timeouts == []


class SlowResetBrain(FakeBrain):
    async def maybe_reset(self) -> None:
        await asyncio.sleep(0.1)


async def test_working_tone_counts_slow_reconnect():
    cfg = SimpleNamespace(followup_s=8.0, working_tone_after_s=0.01)
    loop, d = make([AUDIO], ["slow reconnect"], SlowResetBrain([["Done."]]), cfg=cfg)
    await loop.run()
    assert d.player.played == ["tone:wake", "tone:working", "Done."]


class ExplodingRecorder(FakeRecorder):
    async def record(self, frames, start_timeout_s):
        self.timeouts.append(start_timeout_s)
        raise OSError("mic gone")


async def test_recorder_exception_plays_error_and_loop_continues():
    loop, d = make([], [], FakeBrain(), wakes=2)
    loop.recorder = ExplodingRecorder([])
    await loop.run()
    assert d.player.played == ["tone:wake", "tone:error", "Sorry, something broke: OSError.",
                               "tone:wake", "tone:error", "Sorry, something broke: OSError."]


class SpeakThenTimeoutBrain(FakeBrain):
    async def ask(self, text):
        self.asked.append(text)
        yield "First part. "
        yield "More"
        raise BrainTimeout("slow")


async def test_timeout_after_a_spoken_sentence():
    brain = SpeakThenTimeoutBrain()
    loop, d = make([AUDIO, AUDIO], ["long job"], brain)
    await loop.run()
    assert d.player.played == ["tone:wake", "First part.", "tone:error",
                               "Sorry, that took too long."]
    assert d.recorder.timeouts == [6]  # straight to IDLE, no follow-up window
    assert d.log.turns == [("long job", "First part. Sorry, that took too long.")]


class ClosingBrain(FakeBrain):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def ask(self, text):
        try:
            yield "First sentence. "
            yield "Second sentence."
        finally:
            self.closed = True  # the real brain recycles its client here


class BrokenTTS(FakeTTS):
    def synth(self, text):
        raise RuntimeError("tts died")


async def test_brain_stream_closed_when_speaking_fails():
    brain = ClosingBrain()
    loop, d = make([], [], brain)
    loop.tts = BrokenTTS()
    loop.stt = FakeSTT(["what's up"])
    closed_at_error_tone = []
    play = d.player.play

    async def spy(samples, sr):
        closed_at_error_tone.append(brain.closed)
        await play(samples, sr)

    d.player.play = spy
    await loop.handle_turn(AUDIO)
    # Closed deterministically when the turn failed, not later at GC time.
    assert d.player.played[0] == "tone:error"
    assert closed_at_error_tone[0] is True


async def test_aclose_closes_the_shared_frames_iterator():
    closed = []

    class EndlessMic(FakeMic):
        async def frames(self):
            try:
                while True:
                    yield np.zeros(1280, dtype=np.int16)
            finally:
                closed.append(True)

    loop, d = make([], [], FakeBrain())
    loop.mic = EndlessMic()
    await loop.aclose()  # never started: nothing to close
    frames = loop._stream()
    await frames.__anext__()
    await loop.aclose()
    await loop.aclose()  # idempotent
    assert closed == [True]


async def test_confirm_lets_claude_judge_a_loose_yes():
    judge = fake_judge("APPROVE")
    brain = ConfirmingBrain()
    loop, d = make([AUDIO, AUDIO], ["email sam", "yea approoved"], brain, judge=judge)
    brain.confirm = loop.confirm
    await asyncio.wait_for(loop.run(), timeout=2)
    assert judge.calls == ["yea approoved"]
    assert d.player.played == ["tone:wake", "Send it?", "Sent."]


async def test_loose_nevermind_is_a_fast_cancel():
    brain = FakeBrain([["unused"]])
    loop, d = make([AUDIO], ["ah nevermind then"], brain)
    await loop.run()
    assert brain.asked == []
    assert d.player.played == ["tone:wake", "Okay."]


async def test_long_sentence_mentioning_nevermind_goes_to_claude():
    brain = FakeBrain([["Sure."]])
    text = "what does the word nevermind mean in that old song lyric"
    loop, d = make([AUDIO], [text], brain)
    await loop.run()
    assert brain.asked == [text]


async def test_claude_reading_a_brush_off_ends_the_conversation():
    brain = FakeBrain([["Okay."], ["unused"]])
    loop, d = make([AUDIO, AUDIO], ["eh actually it's fine", "this should not be heard"], brain)
    await loop.run()
    assert brain.asked == ["eh actually it's fine"]
    assert d.recorder.timeouts == [6]  # no follow-up window opened


@pytest.mark.parametrize("phrase", ["Thank you.", "thanks jarvis", "Sounds good.", "cool, thanks",
                                    "perfect thank you so much", "alright that's all", "got it"])
async def test_closing_phrase_in_followup_ends_it(phrase):
    brain = FakeBrain([["Report is due."], ["unused"]])
    loop, d = make([AUDIO, AUDIO, AUDIO], ["what's due", phrase, "never heard"], brain)
    await loop.run()
    assert brain.asked == ["what's due"]
    assert d.recorder.timeouts == [6, 8.0]  # no second follow-up window
    thanked = "thank" in phrase.lower()
    assert d.player.played == ["tone:wake", "Report is due."] + (["You're welcome, sir."] if thanked else [])


async def test_closing_words_inside_a_request_go_to_claude():
    brain = FakeBrain([["Report is due."], ["Playing."], ["unused"]])
    loop, d = make([AUDIO, AUDIO], ["what's due", "thanks, now play some lofi"], brain)
    await loop.run()
    assert brain.asked == ["what's due", "thanks, now play some lofi"]


async def test_sounds_good_after_a_question_goes_to_claude():
    # "sounds good" answers "Want me to set a timer?"; it is not a goodbye.
    brain = FakeBrain([["Want me to set a timer?"], ["Timer set."], ["unused"]])
    loop, d = make([AUDIO, AUDIO], ["remind me in ten", "sounds good"], brain)
    await loop.run()
    assert brain.asked == ["remind me in ten", "sounds good"]


async def test_claude_youre_welcome_ends_the_conversation():
    brain = FakeBrain([["Report is due."], ["You're welcome, sir."], ["unused"]])
    loop, d = make([AUDIO, AUDIO, AUDIO], ["what's due", "cheers mate, lovely stuff", "never"], brain)
    await loop.run()
    assert brain.asked == ["what's due", "cheers mate, lovely stuff"]
    assert d.recorder.timeouts == [6, 8.0]


async def test_trailing_filler_keeps_listening_and_joins_the_audio():
    brain = FakeBrain(replies=[["Playing it."]])
    loop, d = make([AUDIO, AUDIO], ["can you check out, uh, like-",
                                     "can you check out, uh, like- the jazz instrumentals"],
                   brain)
    await loop.run()
    assert brain.asked == ["can you check out, uh, like- the jazz instrumentals"]
    assert d.recorder.timeouts[:2] == [6, 3.0]  # LISTEN, then the continuation


async def test_trailing_filler_with_no_more_speech_answers_what_was_said():
    brain = FakeBrain(replies=[["What did you want?"]])
    loop, d = make([AUDIO], ["can you check out, um"], brain)
    await loop.run()
    assert brain.asked == ["can you check out, um"]
    assert d.recorder.timeouts == [6, 3.0, 8.0]


async def test_finished_sentence_is_not_extended():
    brain = FakeBrain(replies=[["Done."]])
    loop, d = make([AUDIO], ["play some jazz"], brain)
    await loop.run()
    assert brain.asked == ["play some jazz"]
    assert d.recorder.timeouts == [6, 8.0]


async def test_trailing_off_is_extended_at_most_twice():
    brain = FakeBrain(replies=[["Hm?"]])
    loop, d = make([AUDIO] * 4, ["so uh", "so uh and", "so uh and like", "never reached"],
                   brain)
    await loop.run()
    assert brain.asked[0] == "so uh and like"  # the 4th utterance is a new follow-up
    assert d.recorder.timeouts[:4] == [6, 3.0, 3.0, 8.0]


async def test_capped_recording_is_not_extended():
    brain = FakeBrain(replies=[["Okay."]])
    loop, d = make([AUDIO, AUDIO], ["the TV keeps going and", "x"], brain)
    d.recorder.last_end = SimpleNamespace(reason="max_length")
    await loop.run()
    assert brain.asked == ["the TV keeps going and"]


@pytest.mark.parametrize("text,trails", [
    ("check out, uh, like-", True), ("play the one with", True), ("hmm...", True),
    ("um", True), ("play that", False), ("I like it", False), ("is it like", True),
    ("", False),
    ("What's the weather like?", False), ("What does it look like?", False),
    ("Who is that for?", False), ("What should I listen to?", False),
    ("play the one with.", False), ("Turn it up and!", False),
    ("I was thinking, um,", True), ("check out, uh, like-", True),
    ("play the one with", True), ("so like\u2026", True),
])
def test_trails_off(text, trails):
    from jarvis.voice.loop import _trails_off
    assert _trails_off(text) is trails


class FakeDucker:
    def __init__(self, player=None, fail=False) -> None:
        self.events: list[str] = []
        self.player = player
        self.fail = fail

    def duck(self) -> None:
        self.events.append("duck")
        if self.fail:
            raise RuntimeError("osascript died")

    def restore(self) -> None:
        self.events.append("restore")
        if self.fail:
            raise RuntimeError("osascript died")


def make_ducked(utterances, transcripts, brain, ducker):
    loop, d = make(utterances, transcripts, brain)
    loop.ducker = ducker
    return loop, d


async def test_ducked_for_turn_and_restored_after():
    brain = FakeBrain([["Report is due."]])
    ducker = FakeDucker()
    loop, d = make_ducked([AUDIO], ["what's due"], brain, ducker)
    await loop.run()
    assert ducker.events[0] == "duck"
    assert ducker.events[-1] == "restore"
    assert ducker.events.count("restore") == 1


async def test_reducked_before_each_sentence():
    brain = FakeBrain([["One. Two."]])
    ducker = FakeDucker()
    loop, d = make_ducked([AUDIO], ["hi there"], brain, ducker)
    await loop.run()
    # wake duck + one per sentence (One., Two.) then the final restore
    assert ducker.events == ["duck", "duck", "duck", "restore"]


async def test_restored_when_turn_raises():
    brain = FakeBrain(error=RuntimeError("boom"))
    ducker = FakeDucker()
    loop, d = make_ducked([AUDIO], ["what's due"], brain, ducker)
    await loop.run()
    assert ducker.events[-1] == "restore"


async def test_restored_when_conversation_cancelled():
    brain = FakeBrain(delay=5)
    ducker = FakeDucker()
    loop, d = make_ducked([AUDIO], ["what's due"], brain, ducker)
    task = asyncio.create_task(loop.run())
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ducker.events[-1] == "restore"


async def test_ducker_failure_never_fails_the_turn():
    brain = FakeBrain([["Report is due."]])
    ducker = FakeDucker(fail=True)
    loop, d = make_ducked([AUDIO], ["what's due"], brain, ducker)
    await loop.run()
    assert d.player.played == ["tone:wake", "Report is due."]


class HungDucker(FakeDucker):
    """duck() blocks like osascript waiting on a hung Spotify."""

    def __init__(self, player, block_s=0.6) -> None:
        super().__init__(player)
        self.block_s = block_s
        self.finished_at: list[float] = []

    def duck(self) -> None:
        import time
        super().duck()
        time.sleep(self.block_s)
        self.finished_at.append(time.monotonic())

    def restore(self) -> None:
        import time
        self.events.append(("restore", time.monotonic()))


class TimedPlayer(FakePlayer):
    async def play(self, samples, sr) -> None:
        import time
        await super().play(samples, sr)
        self.times = getattr(self, "times", []) + [time.monotonic()]


async def test_a_hung_music_app_never_holds_up_the_reply():
    brain = FakeBrain([["One. Two. Three."]])
    loop, d = make([AUDIO], ["count"], brain)
    d.player = loop.player = TimedPlayer(d.mic)
    ducker = loop.ducker = HungDucker(d.player)
    await asyncio.wait_for(loop.run(), timeout=5)
    assert d.player.played == ["tone:wake", "One.", "Two.", "Three."]
    # Every sentence played while the first duck was still stuck, ~0.3 s apart at most.
    first_duck_done = ducker.finished_at[0]
    assert d.player.times[2] < first_duck_done
    gaps = [b - a for a, b in zip(d.player.times, d.player.times[1:])]
    assert max(gaps) < 0.45
    # No pile-up of osascript calls behind the stuck one, and restore waits for it.
    assert ducker.events.count("duck") <= 2
    restore_at = ducker.events[-1][1]
    assert restore_at >= max(ducker.finished_at)


# -- barge-in ---------------------------------------------------------------

from jarvis.voice.bargein import BargeIn  # noqa: E402


class FakeBarge:
    """Scripted listener: one entry per listened sentence, a BargeIn or None."""

    def __init__(self, hits=(), delay: float = 0.0) -> None:
        self.hits = list(hits)
        self.delay = delay
        self.speaking: list[str] = []

    async def listen(self, frames, speaking, done):
        self.speaking.append(speaking())
        hit = self.hits.pop(0) if self.hits else None
        if hit is None:
            await done.wait()
            return None
        await asyncio.sleep(self.delay)  # playback has started
        return hit


class BargePlayer(FakePlayer):
    """Plays take a little real time; stop() cuts the clip at `fraction`."""

    def __init__(self, mic: FakeMic, fraction: float = 0.5) -> None:
        super().__init__(mic)
        self.fraction = fraction
        self.stops = 0
        self._stopped = asyncio.Event()

    async def play(self, samples, sr) -> float:
        await super().play(samples, sr)
        self._stopped.clear()
        try:
            await asyncio.wait_for(self._stopped.wait(), 0.05)
        except TimeoutError:
            return 1.0
        return self.fraction

    def stop(self) -> None:
        self.stops += 1
        self._stopped.set()


class InterruptBrain(FakeBrain):
    def __init__(self, replies=None) -> None:
        super().__init__(replies)
        self.events: list[str] = []

    async def interrupt(self) -> None:
        self.events.append("interrupt")

    async def ask(self, text):
        try:
            async for delta in super().ask(text):
                yield delta
        finally:
            self.events.append("closed")


class LengthSTT(FakeSTT):
    def __init__(self, transcripts) -> None:
        super().__init__(transcripts)
        self.lengths: list[int] = []

    def transcribe(self, audio):
        self.lengths.append(len(audio))
        return super().transcribe(audio)


CAPTURED = np.ones(800, dtype=np.int16)
STORY = ["One two three four. Second", " sentence."]
PREFIX = '(You were interrupted after saying: "One two". The rest was not heard.) '


def make_barge(utterances, transcripts, brain, hits=(), fraction=0.5):
    loop, d = make(utterances, transcripts, brain)
    d.player = loop.player = BargePlayer(d.mic, fraction)
    d.barge = loop.barge = FakeBarge(hits)
    d.stt = loop.stt = LengthSTT(transcripts)
    return loop, d


async def test_barge_in_stops_playback_and_the_rest_of_the_reply(capsys):
    brain = InterruptBrain([STORY, ["Dogs are great."]])
    loop, d = make_barge([AUDIO, AUDIO], ["tell me a story", "wait, what about dogs"], brain,
                         hits=[BargeIn("wait", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.stops == 1
    assert d.player.played == ["tone:wake", "One two three four.", "Dogs are great."]
    # Interrupted before the ask was closed, so the brain keeps its session.
    assert brain.events[:2] == ["interrupt", "closed"]
    assert brain.asked == ["tell me a story", PREFIX + "wait, what about dogs"]
    # Only what was actually played is logged; the prefix never reaches the log.
    assert d.log.turns == [("tell me a story", "One two"),
                           ("wait, what about dogs", "Dogs are great.")]
    # The continuation is recorded with the follow-up timeout, joined to the captured audio.
    assert d.recorder.timeouts == [6, 8.0, 8.0]
    assert d.stt.lengths == [len(AUDIO), len(CAPTURED) + len(AUDIO)]
    err = capsys.readouterr().err
    assert "[interrupt] wait" in err
    assert "[heard] wait, what about dogs" in err
    assert "interrupted after saying" not in err


async def test_stop_words_alone_open_the_followup_silently():
    brain = InterruptBrain([STORY, ["Dogs are great."]])
    loop, d = make_barge([AUDIO, AUDIO, AUDIO], ["tell me a story", "hold up, wait", "dogs?"],
                         brain, hits=[BargeIn("hold up", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played == ["tone:wake", "One two three four.", "Dogs are great."]
    assert d.recorder.timeouts == [6, 8.0, 8.0, 8.0]
    # Claude still learns where it was cut off, on the turn that reaches it.
    assert brain.asked == ["tell me a story", PREFIX + "dogs?"]


async def test_jarvis_alone_after_an_interrupt_opens_the_followup_silently():
    brain = InterruptBrain([STORY, ["Sure."]])
    loop, d = make_barge([AUDIO, AUDIO, AUDIO], ["tell me a story", "Jarvis.", "go on"],
                         brain, hits=[BargeIn("jarvis", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played == ["tone:wake", "One two three four.", "Sure."]
    assert brain.asked == ["tell me a story", PREFIX + "go on"]


async def test_cancel_after_an_interrupt_uses_cancel_handling():
    brain = InterruptBrain([STORY, ["unused"]])
    loop, d = make_barge([AUDIO, AUDIO, AUDIO], ["tell me a story", "stop", "never heard"],
                         brain, hits=[BargeIn("stop", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played == ["tone:wake", "One two three four.", "Okay."]
    assert brain.asked == ["tell me a story"]
    assert d.recorder.timeouts == [6, 8.0]


async def test_silence_after_an_interrupt_ends_the_conversation():
    brain = InterruptBrain([STORY])
    loop, d = make_barge([AUDIO], ["tell me a story"], brain, hits=[BargeIn("wait", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.recorder.timeouts == [6, 8.0]
    assert brain.asked == ["tell me a story"]


async def test_reply_plays_unmuted_while_listening_for_stop_words():
    brain = FakeBrain([["One. Two."]])
    loop, d = make_barge([AUDIO], ["count"], brain)
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played == ["tone:wake", "One.", "Two."]
    assert d.player.muted_flags == [True, False, False]  # the wake tone stays muted
    assert d.barge.speaking == ["One.", "Two."]  # echo check sees the current sentence
    # Wake tone, frames queued while thinking dropped once, a 0.3 s tail per sentence.
    assert d.mic.mute_calls == 4
    assert d.player.stops == 0


async def test_barge_none_keeps_every_playback_muted():
    brain = FakeBrain([["One. Two."]])
    loop, d = make([AUDIO], ["count"], brain)
    assert loop.barge is None
    await loop.run()
    assert all(d.player.muted_flags)


async def test_confirmation_questions_are_never_barge_in_able():
    brain = ConfirmingBrain()
    loop, d = make_barge([AUDIO, AUDIO], ["email Sam", "yes"], brain)
    brain.confirm = loop.confirm
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played == ["tone:wake", "Send it?", "Sent."]
    assert d.player.muted_flags == [True, True, False]
    assert d.barge.speaking == ["Sent."]


async def test_error_messages_are_never_barge_in_able():
    brain = FakeBrain(error=RuntimeError("boom"))
    loop, d = make_barge([AUDIO], ["what's due"], brain)
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played[1:] == ["tone:error", "Sorry, something broke: RuntimeError."]
    assert all(d.player.muted_flags)
    assert d.barge.speaking == []


async def test_music_restored_after_an_interrupted_conversation():
    brain = InterruptBrain([STORY, ["Dogs are great."]])
    loop, d = make_barge([AUDIO, AUDIO], ["tell me a story", "wait, dogs"], brain,
                         hits=[BargeIn("wait", CAPTURED)])
    ducker = loop.ducker = FakeDucker()
    await asyncio.wait_for(loop.run(), timeout=2)
    assert ducker.events[0] == "duck"
    assert ducker.events[-1] == "restore"
    assert ducker.events.count("restore") == 1


async def test_interrupted_before_any_word_was_heard():
    brain = InterruptBrain([STORY, ["Okay, dogs."]])
    loop, d = make_barge([AUDIO, AUDIO], ["tell me a story", "no, dogs"], brain,
                         hits=[BargeIn("no", CAPTURED)], fraction=0.0)
    await asyncio.wait_for(loop.run(), timeout=2)
    assert brain.asked[1] == ("(You were interrupted before saying anything. "
                              "The rest was not heard.) no, dogs")
    assert d.log.turns[0] == ("tell me a story", "")


class BrokenBarge(FakeBarge):
    async def listen(self, frames, speaking, done):
        raise RuntimeError("vad died")


async def test_listener_failure_never_fails_the_reply():
    brain = FakeBrain([["One. Two."]])
    loop, d = make_barge([AUDIO], ["count"], brain)
    loop.barge = BrokenBarge()
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played == ["tone:wake", "One.", "Two."]
    assert d.log.turns == [("count", "One. Two.")]


class DetachedConfirmBrain(InterruptBrain):
    """The SDK's permission callback runs detached: it queues on the audio lock
    while the sentence before the gated tool call is still playing."""

    def __init__(self, replies=None) -> None:
        super().__init__(replies)
        self.pending: list[asyncio.Task] = []
        self.speaking_lock = None

    async def ask(self, text):
        if not self.asked:
            self.pending.append(asyncio.create_task(self._confirm_while_speaking()))
        async for delta in super().ask(text):
            yield delta

    async def _confirm_while_speaking(self):
        while self.speaking_lock is None or not self.speaking_lock.locked():
            await asyncio.sleep(0)
        return await self.confirm("Send it?")


async def test_confirm_queued_behind_an_interrupted_sentence_is_denied_silently():
    brain = DetachedConfirmBrain([["Sure, I'll email Sam. Then", " more."], ["Emailing Bob."]])
    loop, d = make_barge([AUDIO, AUDIO], ["email Sam", "no, email Bob instead"], brain,
                         hits=[BargeIn("no", CAPTURED)])
    brain.confirm = loop.confirm
    brain.speaking_lock = loop._audio_lock
    d.barge.delay = 0.01  # the user says "no" a moment into the sentence
    await asyncio.wait_for(loop.run(), timeout=2)
    assert [await t for t in brain.pending] == [False]
    assert "Send it?" not in d.player.played
    assert d.player.played[-1] == "Emailing Bob."
    # The continuation went to Claude, not to the confirmation.
    assert brain.asked[1].endswith("no, email Bob instead")
    assert d.recorder.timeouts == [6, 8.0, 8.0]


STOP_FRAME = np.full(1280, 7, dtype=np.int16)


class QueueMic(FakeMic):
    """A queued mic: mute() drains what is pending, like MicStream."""

    def __init__(self) -> None:
        super().__init__(0)
        self.pending: list[np.ndarray] = []

    async def frames(self):
        while True:
            await asyncio.sleep(0.001)
            yield self.pending.pop(0) if self.pending else None  # None: nothing yet

    @contextmanager
    def mute(self, seconds_tail: float = 0.3):
        self.pending.clear()
        with super().mute(seconds_tail):
            yield
        self.pending.clear()


class FrameBarge:
    """Reads the real frames iterator; STOP_FRAME is the user saying "wait"."""

    async def listen(self, frames, speaking, done):
        while not done.is_set():
            frame = await anext(frames)
            if frame is not None and frame[0] == 7:
                return BargeIn("wait", CAPTURED)
        return None


class QueueingBrain(InterruptBrain):
    """Streams a reply, queueing mic frames (and moving the clock) between sentences."""

    def __init__(self, script, mic, clock=None) -> None:
        super().__init__()
        self.script, self.mic, self.clock = script, mic, clock

    async def ask(self, text):
        self.asked.append(text)
        try:
            for step in self.script:
                if step == "stop":
                    self.mic.pending.append(STOP_FRAME)
                elif isinstance(step, float):
                    self.clock[0] += step
                else:
                    yield step
        finally:
            self.events.append("closed")


def make_queued(script, advance_clock=False):
    loop, d = make_barge([], ["tell me"], None)
    mic = loop.mic = d.mic = QueueMic()
    d.player = loop.player = BargePlayer(mic)
    loop.barge = FrameBarge()
    clock = [100.0]
    loop._clock = lambda: clock[0]
    brain = loop.brain = QueueingBrain(script, mic, clock)
    return loop, d, brain


async def test_stop_word_between_sentences_cuts_the_next_one():
    loop, d, brain = make_queued(["One two. Three", "stop", " four. Five", " six."])
    await asyncio.wait_for(loop.handle_turn(AUDIO), timeout=2)
    await loop.aclose()
    assert brain.events[:2] == ["interrupt", "closed"]
    assert d.player.played == ["One two.", "Three four."]


async def test_audio_from_the_thinking_phase_is_dropped():
    loop, d, brain = make_queued(["stop", "One two. Three", " four."])
    await asyncio.wait_for(loop.handle_turn(AUDIO), timeout=2)
    await loop.aclose()
    assert "interrupt" not in brain.events
    assert d.player.played == ["One two.", "Three four."]


async def test_audio_from_a_long_pause_between_sentences_is_dropped():
    # A tool call between sentences: what was queued meanwhile is not a reply to Jarvis.
    loop, d, brain = make_queued(["One two. Three", "stop", 1.5, " four."])
    await asyncio.wait_for(loop.handle_turn(AUDIO), timeout=2)
    await loop.aclose()
    assert "interrupt" not in brain.events
    assert d.player.played == ["One two.", "Three four."]


@pytest.mark.parametrize("text,cancels", [
    ("wait, never mind", True), ("ah nevermind then", True), ("nevermind, forget it", True),
    ("no wait, never mind, play rock", False), ("cancel that timer", False),
])
def test_cancel_word_only_cancels_with_nothing_after_it(text, cancels):
    from jarvis.voice.loop import _is_cancel
    assert _is_cancel(text) is cancels


async def test_never_mind_then_a_new_request_after_an_interrupt_goes_to_claude():
    brain = InterruptBrain([STORY, ["Rock it is."]])
    loop, d = make_barge([AUDIO, AUDIO], ["tell me a story", "no wait, never mind, play rock"],
                         brain, hits=[BargeIn("no", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert brain.asked[1] == PREFIX + "no wait, never mind, play rock"
    assert d.player.played[-1] == "Rock it is."


# The listener's VAD hears Jarvis's own voice too, so the audio it captured (and the
# continuation transcript) starts with the words played before the stop phrase.

async def test_echoed_words_then_wait_open_the_followup_silently():
    brain = InterruptBrain([STORY, ["Dogs are great."]])
    loop, d = make_barge([AUDIO, AUDIO, AUDIO], ["tell me a story", "One two wait.", "dogs?"],
                         brain, hits=[BargeIn("wait", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played == ["tone:wake", "One two three four.", "Dogs are great."]
    assert brain.asked == ["tell me a story", PREFIX + "dogs?"]


async def test_echoed_words_then_stop_cancel():
    brain = InterruptBrain([STORY, ["unused"]])
    loop, d = make_barge([AUDIO, AUDIO, AUDIO], ["tell me a story", "One two three stop",
                                                 "never heard"],
                         brain, hits=[BargeIn("stop", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert d.player.played == ["tone:wake", "One two three four.", "Okay."]
    assert brain.asked == ["tell me a story"]


async def test_echoed_words_are_trimmed_before_claude_and_the_log(capsys):
    brain = InterruptBrain([STORY, ["Jazz it is."]])
    loop, d = make_barge([AUDIO, AUDIO], ["tell me a story", "One two, wait, play jazz instead"],
                         brain, hits=[BargeIn("wait", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert brain.asked[1] == PREFIX + "wait, play jazz instead"
    assert d.log.turns[1] == ("wait, play jazz instead", "Jazz it is.")
    err = capsys.readouterr().err
    assert "[heard] wait, play jazz instead" in err
    assert "[heard] One two" not in err


async def test_echo_without_the_heard_phrase_strips_the_played_words():
    # Whisper heard the stop phrase in its window but not in the full continuation.
    brain = InterruptBrain([STORY, ["Jazz it is."]])
    loop, d = make_barge([AUDIO, AUDIO], ["tell me a story", "One two three, play jazz"],
                         brain, hits=[BargeIn("hold up", CAPTURED)])
    await asyncio.wait_for(loop.run(), timeout=2)
    assert brain.asked[1] == PREFIX + "play jazz"


def test_trim_echo():
    from jarvis.voice.loop import _trim_echo
    said = "The weather today is sunny."
    assert _trim_echo("The weather today is sunny and wait", "wait", said) == "wait"
    assert _trim_echo("weather today, hold up, play jazz", "hold up", said) == "hold up, play jazz"
    assert _trim_echo("The weather today play jazz", "wait", said) == "play jazz"
    assert _trim_echo("play jazz", "wait", said) == "play jazz"
    assert _trim_echo("The weather today is sunny.", "wait", said) == ""
    assert _trim_echo("Hey Jarvis, play jazz", "jarvis", said) == "Jarvis, play jazz"
