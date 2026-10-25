from pathlib import Path

import numpy as np
import pytest

from jarvis.voice.speech import CODE_SPOKEN, KokoroTTS, SentenceSplitter, to_speakable

MODEL = Path("models/kokoro/kokoro-v1.0.onnx")
VOICES = Path("models/kokoro/voices-v1.0.bin")


def test_strips_markdown():
    out = to_speakable("**Due** tomorrow:\n- Report")
    assert "*" not in out and "-" not in out
    assert out == "Due tomorrow: Report"


def test_strips_headers_numbers_inline_code():
    out = to_speakable("# Plan\n1. Run `make test`\n2. *Ship* it")
    assert out == "Plan Run make test Ship it"


def test_code_fence_replaced():
    out = to_speakable("Try this:\n```python\nprint('x')\n```\nDone.")
    assert "print" not in out and "`" not in out
    assert "I put the code in the log." in out


def test_unclosed_fence_replaced():
    assert to_speakable("Look: ```rm -rf") == "Look: I put the code in the log."


def test_link_label_kept():
    assert to_speakable("See [the docs](https://example.com/a) now") == "See the docs now"


def test_bare_url_becomes_a_link():
    assert to_speakable("Go to https://example.com/x?y=1.") == "Go to a link."


def test_splitter_streams():
    s = SentenceSplitter()
    out = s.feed("Hi the") + s.feed("re. Report is du") + s.feed("e. Ok")
    assert out == ["Hi there.", "Report is due."]
    assert s.flush() == ["Ok"]
    assert s.flush() == []


def test_splitter_keeps_decimals():
    s = SentenceSplitter()
    assert s.feed("It's 3.") == []
    assert s.feed("5 hours.") == []
    assert s.flush() == ["It's 3.5 hours."]


def test_splitter_abbreviations():
    s = SentenceSplitter()
    out = s.feed("Ask Dr. Lee, e.g. tomorrow. Fine! ")
    assert out == ["Ask Dr. Lee, e.g. tomorrow.", "Fine!"]


def test_splitter_newline_and_list_numbers():
    s = SentenceSplitter()
    out = s.feed("1. First\n2. Second\n")
    assert out == ["1. First", "2. Second"]


def test_splitter_holds_open_fence():
    s = SentenceSplitter()
    out = s.feed("Here. Code: ```py\nx = 1. y\n\n")
    assert out == ["Here."]
    out = s.feed("z\n```\nDone. ")
    assert len(out) == 2
    assert out[0].startswith("Code:") and out[0].endswith("```")
    assert out[1] == "Done."
    assert to_speakable(out[0]) == "Code: I put the code in the log."


def _speak_stream(deltas):
    sp = SentenceSplitter()
    chunks = [c for d in deltas for c in sp.feed(d)] + sp.flush()
    return [t for t in (to_speakable(c) for c in chunks) if t]


def test_bold_across_sentence_boundary():
    out = _speak_stream(["**Heads up. Report is due** tomor", "row. "])
    assert out == ["Heads up.", "Report is due tomorrow."]


def test_inline_code_across_sentence_boundary():
    assert _speak_stream(["Say `a. b` ok."]) == ["Say a.", "b ok."]


def test_arithmetic_star_kept():
    assert to_speakable("3 * 4 is 12") == "3 * 4 is 12"


def test_blockquote_and_rule():
    assert to_speakable("> a quote\n---\nafter") == "a quote after"


def test_table_rows():
    out = to_speakable("| a | b |\n|---|---|\n| 1 | 2 |")
    assert out == "a, b 1, 2"


def test_strikethrough_and_tilde_fence():
    assert to_speakable("~~old~~ new") == "old new"
    assert to_speakable("x\n~~~\ncode\n~~~\ny") == f"x {CODE_SPOKEN} y"


def test_splitter_holds_tilde_fence():
    sp = SentenceSplitter()
    assert sp.feed("A. ~~~\nb. c\n") == ["A."]
    out = sp.feed("~~~\nD. ")
    assert out[0].endswith("~~~") and out[1] == "D."


def test_autolink_angle_brackets():
    assert to_speakable("See <https://x.com/a> now") == "See a link now"


@pytest.mark.skipif(not (MODEL.exists() and VOICES.exists()), reason="kokoro model absent")
def test_kokoro_smoke():
    audio, sr = KokoroTTS(MODEL, VOICES).synth("Hello there.")
    assert sr == 24000
    assert audio.size > 0 and np.issubdtype(audio.dtype, np.floating)
