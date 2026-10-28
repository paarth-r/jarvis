#!/usr/bin/env python3
"""Offline false-barge-in soak: does Jarvis's own voice ever trigger a stop phrase?

Synthesizes assistant replies with the real KokoroTTS from your voice config, splits
them into sentences the way the loop does, and feeds each sentence's audio (16 kHz
int16, 1280-sample frames) through the real BargeInListener (Silero VAD, a Whisper
with no vocabulary prompt, the configured stop phrases, step_s=0.25, two-window
confirmation) with
`speaking` returning that sentence. Every hit is a false barge-in: nobody spoke.

This feeds the TTS output straight in. It does not model the room (speaker, mic,
reverb, music playing): treat zero hits here as necessary, not sufficient.

    .venv-voice/bin/python scripts/bargein_soak.py [--noise 0.01] [--gain 0.5]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time

import numpy as np

from jarvis.voice.bargein import BargeInListener, find_stop_phrase
from jarvis.voice.config import load_config
from jarvis.voice.main import DEFAULT_CONFIG, DEFAULT_LOCAL, resolve_config_path, resolve_paths
from jarvis.voice.speech import KokoroTTS, SentenceSplitter, to_speakable
from jarvis.voice.stt import WhisperSTT
from jarvis.voice.turn import FRAME, SAMPLE_RATE, SileroVAD

# Near-misses for every default stop phrase: words the echo rule removes when they are
# spoken literally, and words Whisper might mishear as a stop phrase in a cut window.
REPLIES = [
    "Right now it's sixty-two degrees and sunny. I know you wanted rain, but nothing is "
    "forecast until Thursday.",
    "You have nothing due tomorrow. Your next deadline is not until next week, so you're in "
    "good shape for now.",
    "I'd notice the wait time at that place is about forty minutes. Want me to find "
    "somewhere closer?",
    "Done, the timer is set for ten minutes. I'll let you know when it goes off.",
    "Playing a jazz album on Spotify now. Turn it up if you'd like it louder.",
    "The meeting is on hold until Friday. Nobody has moved it again since this morning.",
    "I'm Jarvis, your voice assistant. I can play music, check your calendar, and keep notes.",
    "Your package is now out for delivery. It should arrive before noon, though not "
    "before ten.",
    "Hang the picture about five feet up. That keeps it at eye level for most people.",
    "Stopping at the store on the way home adds about six minutes. Shall I add a reminder?",
    "Known issue: the build fails on the Orin when the cache is stale. Running a clean "
    "build usually fixes it.",
    "The answer is no longer seventeen, it's nineteen. I rechecked the second step.",
    "Never mind the old schedule. Practice is now at four, not three, starting Monday.",
    "Wait times at the DMV are long today. The earliest open slot is at two fifteen.",
    "Hold the club a little lower and keep your weight on the front foot. That should fix "
    "the slice.",
    "Noted. I added milk, eggs, and bread to your shopping list.",
    "The Warriors won, one hundred twelve to one oh eight. Curry had thirty-one points.",
    "Your heart rate held steady at fifty-eight overnight. Sleep was about seven hours.",
    "I don't know that one yet. Do you want me to look it up, or just make a note?",
    "Notifications are off until morning. I'll stay quiet unless you call me.",
    "Okay, now playing your Liked Songs on shuffle. The first one is Nights by Frank Ocean.",
    "There's no rain today, and the wind is light. It's a good afternoon for the range.",
]


def sentences(reply: str) -> list[str]:
    splitter = SentenceSplitter()
    out = list(splitter.feed(reply)) + list(splitter.flush())
    return [s for s in (to_speakable(x) for x in out) if s]


def to_frames(audio: np.ndarray, sr: int, gain: float, noise: float,
              rng: np.random.Generator) -> list[np.ndarray]:
    from scipy.signal import resample_poly

    g = np.gcd(SAMPLE_RATE, sr)
    x = resample_poly(np.asarray(audio, np.float32), SAMPLE_RATE // g, sr // g) * gain
    tail = np.zeros(SAMPLE_RATE // 2, np.float32)  # half a second of quiet ends the segment
    x = np.concatenate([x, tail])
    if noise:
        x = x + rng.normal(0.0, noise, len(x)).astype(np.float32)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    n = len(pcm) // FRAME
    return [pcm[i * FRAME:(i + 1) * FRAME] for i in range(n)]


class LoggingSTT:
    """Records every window transcript the listener sees."""

    def __init__(self, stt: WhisperSTT) -> None:
        self._stt, self.texts = stt, []

    def transcribe(self, audio: np.ndarray) -> str | None:
        text = self._stt.transcribe(audio)
        self.texts.append(text or "")
        return text


async def feed(frames: list[np.ndarray]):
    for f in frames:
        yield f


async def soak(args: argparse.Namespace) -> int:
    cfg = resolve_paths(load_config(resolve_config_path(DEFAULT_CONFIG),
                                    resolve_config_path(DEFAULT_LOCAL)))
    phrases = list(args.phrases or cfg.stop_phrases)
    tts = KokoroTTS(cfg.kokoro_model, cfg.kokoro_voices, cfg.tts_voice)
    stt = LoggingSTT(WhisperSTT(cfg.stt_model))
    stt.transcribe(np.zeros(SAMPLE_RATE, np.int16))  # load the weights
    stt.texts.clear()
    listener = BargeInListener(SileroVAD(), stt, phrases, step_s=0.25,
                               confirm_windows=1 if args.single_window else 2)
    rng = np.random.default_rng(0)
    total = hits = 0
    fired: dict[str, int] = {}
    near: list[str] = []
    start = time.monotonic()
    for reply in REPLIES:
        for sentence in sentences(reply):
            total += 1
            samples, sr = tts.synth(sentence)
            frames = to_frames(samples, sr, args.gain, args.noise, rng)
            before = len(stt.texts)
            hit = await listener.listen(feed(frames), lambda s=sentence: s, asyncio.Event())
            windows = stt.texts[before:]
            # Windows that contained a stop phrase the echo rule then threw away.
            for w in windows:
                if find_stop_phrase(w, "", phrases) and not find_stop_phrase(w, sentence, phrases):
                    near.append(f"{w!r} (echo of {sentence!r})")
            if hit is not None:
                hits += 1
                fired[hit.phrase] = fired.get(hit.phrase, 0) + 1
                print(f"HIT {hit.phrase!r}: speaking {sentence!r}; windows {windows!r}")
    took = time.monotonic() - start
    print(f"\n{total} sentences, {len(REPLIES)} replies, gain={args.gain} noise={args.noise}, "
          f"{len(stt.texts)} STT windows, {took:.0f} s")
    print(f"false barge-ins: {hits} {fired or ''}")
    print(f"stop phrases removed by the echo rule: {len(near)}")
    for line in near[:args.show]:
        print(f"  {line}")
    return 1 if hits else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gain", type=float, default=1.0, help="scale the TTS audio (default 1.0)")
    p.add_argument("--noise", type=float, default=0.0,
                   help="std of added white noise, full scale = 1.0 (try 0.005-0.02)")
    p.add_argument("--phrases", nargs="*", help="stop phrases (default: from config)")
    p.add_argument("--single-window", action="store_true",
                   help="fire on one STT window (the behaviour before the two-window check)")
    p.add_argument("--show", type=int, default=20, help="echo-rule removals to print")
    return asyncio.run(soak(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
