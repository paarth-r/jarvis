import platform
import subprocess
import sys
import wave

import numpy as np
import pytest

from jarvis.voice.stt import (
    WhisperSTT,
    clean_transcript,
    load_vocab_prompt,
    strip_wake_phrase,
)


def _result(text, *probs):
    return {"text": text, "segments": [{"no_speech_prob": p} for p in probs]}


def test_phantom_dropped():
    assert clean_transcript(_result("Thank you for watching.", 0.1)) is None


def test_high_no_speech_dropped():
    assert clean_transcript(_result("Turn on the fan.", 0.7, 0.9)) is None


def test_real_text_kept():
    assert clean_transcript(_result(" Turn on the fan. ", 0.1)) == "Turn on the fan."


def test_empty_dropped():
    assert clean_transcript(_result("   ")) is None


def test_mixed_no_speech_kept():
    assert clean_transcript(_result("Turn on the fan.", 0.9, 0.1)) == "Turn on the fan."


def test_vocab_prompt_format(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("# comment\n\nwidgetPose\n  Smith  \n")
    assert load_vocab_prompt(p) == "Vocabulary: widgetPose, Smith."


def test_vocab_missing_none(tmp_path):
    assert load_vocab_prompt(tmp_path / "nope.txt") is None
    assert load_vocab_prompt(None) is None


def test_vocab_empty_none(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("# only comment\n")
    assert load_vocab_prompt(p) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hey Jarvis, turn on the fan.", "turn on the fan."),
        ("hey claude what's due", "what's due"),
        ("Hey Jarvis. What time is it", "What time is it"),
        ("Jarvis is cool", "Jarvis is cool"),
    ],
)
def test_strip_wake_phrase(text, expected):
    assert strip_wake_phrase(text) == expected


@pytest.mark.skipif(
    not (sys.platform == "darwin" and platform.machine() == "arm64"),
    reason="mlx-whisper needs Apple silicon",
)
def test_whisper_smoke(tmp_path):
    wav = tmp_path / "say.wav"
    subprocess.run(
        ["say", "-o", str(wav), "--data-format=LEI16@16000",
         "add tennis to my calendar on Sunday"],
        check=True,
    )
    with wave.open(str(wav)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    stt = WhisperSTT("mlx-community/whisper-large-v3-turbo", None)
    text = stt.transcribe(audio)
    assert text is not None
    assert "tennis" in text.lower() and "sunday" in text.lower()
