"""Speakable text: markup stripping, streaming sentence splitting, Kokoro TTS."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import numpy as np

CODE_SPOKEN = "I put the code in the log."
FENCE = "```"
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "vs", "e.g", "i.e", "etc"}

_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"https?://\S*[^\s.,;:!?)\]]")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)\*(?![\w*])")
_ITALIC_US_RE = re.compile(r"(?<![\w_])_(?!\s)([^_\n]+?)_(?![\w_])")


def to_speakable(text: str) -> str:
    """Strip markdown/code/URLs so the TTS never reads out markup. May return ""."""
    text = _FENCE_RE.sub(f" {CODE_SPOKEN} ", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub("a link", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _HEADER_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _ITALIC_US_RE.sub(r"\1", text)
    return " ".join(text.split())


class SentenceSplitter:
    """Incremental splitter for streamed text deltas.

    Splits on `. ! ?` followed by whitespace, or on newline. Never splits
    decimals, common abbreviations, or list numbers, and holds an open ```
    fence so the whole fence is emitted as one chunk once it closes.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        start = 0
        i = 0
        buf = self._buf
        while i < len(buf):
            if buf.startswith(FENCE, i):
                close = buf.find(FENCE, i + len(FENCE))
                if close == -1:
                    break
                i = close + len(FENCE)
                continue
            c = buf[i]
            if c == "\n" or (c in ".!?" and self._is_boundary(buf, start, i)):
                sentence = buf[start : i + 1].strip()
                if sentence:
                    out.append(sentence)
                start = i + 1
            i += 1
        self._buf = buf[start:]
        return out

    @staticmethod
    def _is_boundary(buf: str, start: int, i: int) -> bool:
        if i + 1 >= len(buf) or not buf[i + 1].isspace():
            return False
        if buf[i] != ".":
            return True
        so_far = buf[start:i].strip()
        if so_far.isdigit():  # list number like "1."
            return False
        words = so_far.split()
        last = words[-1].lower().lstrip("(\"'") if words else ""
        return last not in _ABBREVIATIONS

    def flush(self) -> list[str]:
        rest = self._buf.strip()
        self._buf = ""
        return [rest] if rest else []


class TTS(Protocol):
    def synth(self, text: str) -> tuple[np.ndarray, int]: ...


class KokoroTTS:
    def __init__(self, model: Path, voices: Path, voice: str = "af_heart") -> None:
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(str(model), str(voices))
        self._voice = voice

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        audio, sr = self._kokoro.create(text, voice=self._voice, speed=1.0, lang="en-us")
        return audio, sr
