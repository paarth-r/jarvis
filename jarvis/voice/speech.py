"""Speakable text: markup stripping, streaming sentence splitting, Kokoro TTS."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

import numpy as np

CODE_SPOKEN = "I put the code in the log."
FENCES = ("```", "~~~")
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "vs", "e.g", "i.e", "etc"}

_FENCE_RE = re.compile(r"(```|~~~).*?(?:\1|\Z)", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"<?https?://[^\s<>]*[^\s<>.,;:!?)\]]>?")
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_QUOTE_RE = re.compile(r"^[ \t]*>+[ \t]?", re.MULTILINE)
_RULE_RE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$", re.MULTILINE)
_TABLE_SEP_RE = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-*:?[ \t]*)+$\n?", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|(.*?)\|?[ \t]*$", re.MULTILINE)
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_LEADING_HASH_RE = re.compile(r"^[ \t]*#+", re.MULTILINE)
_STRAY_STAR_RE = re.compile(r"(?<=\d)\s?\*\s?(?=\d)|\*")
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
    text = _QUOTE_RE.sub("", text)
    text = _RULE_RE.sub("", text)
    text = _TABLE_SEP_RE.sub("", text)
    text = _TABLE_ROW_RE.sub(
        lambda m: ", ".join(c.strip() for c in m.group(1).split("|") if c.strip()), text
    )
    text = _STRIKE_RE.sub(r"\1", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _HEADER_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _ITALIC_US_RE.sub(r"\1", text)
    # Leftovers from markup split across chunks (e.g. bold spanning a sentence end).
    text = _STRAY_STAR_RE.sub(lambda m: m.group(0) if len(m.group(0)) > 1 else "", text)
    text = _LEADING_HASH_RE.sub("", text)
    text = text.replace("`", "").replace("~", "")
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
            fence = next((f for f in FENCES if buf.startswith(f, i)), None)
            if fence:
                close = buf.find(fence, i + 3)
                if close == -1:
                    break
                i = close + 3
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
    def __init__(self, model: Path, voices: Path, voice: str = "bm_daniel") -> None:
        from kokoro_onnx import Kokoro

        self._kokoro = Kokoro(str(model), str(voices))
        self._voice = voice
        # Kokoro voice ids start with "b" for British English, "a" for American.
        self._lang = "en-gb" if voice.startswith("b") else "en-us"

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        audio, sr = self._kokoro.create(text, voice=self._voice, speed=1.0, lang=self._lang)
        return audio, sr
