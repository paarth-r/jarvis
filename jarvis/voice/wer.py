"""Word error rate over the utterance archive (`<data_dir>/utterances.jsonl`)."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_NON_WORD = re.compile(r"[^a-z0-9' ]+")


def _words(text: str | None) -> list[str]:
    return _NON_WORD.sub(" ", (text or "").lower()).split()


def _align(ref: list[str], hyp: list[str]) -> tuple[int, list[str]]:
    """Return (edit distance, reference words that were substituted or deleted)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    missed: list[str] = []
    i, j = n, m
    while i > 0:
        if j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                missed.append(ref[i - 1])
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            missed.append(ref[i - 1])
            i -= 1
        else:
            j -= 1
    missed.reverse()
    return d[n][m], missed


def wer(ref: str, hyp: str) -> float:
    """Word-level Levenshtein distance over reference word count."""
    r, h = _words(ref), _words(hyp)
    if not r:
        return 0.0 if not h else float(len(h))
    return _align(r, h)[0] / len(r)


def missed_words(ref: str, hyp: str) -> list[str]:
    """Reference words the hypothesis got wrong or dropped."""
    return _align(_words(ref), _words(hyp))[1]


@dataclass
class ArchiveScore:
    wer: float
    count: int
    top_missed: list[tuple[str, int]]
    skipped: int = 0


def score_archive(data_dir: Path, top: int = 10) -> ArchiveScore:
    """Score every row with `corrected` set; a null transcript is all deletions."""
    path = Path(data_dir).expanduser() / "utterances.jsonl"
    errors = words = count = skipped = 0
    missed: Counter[str] = Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(row, dict):
            skipped += 1
            continue
        if not row.get("corrected"):
            continue
        ref, hyp = _words(row["corrected"]), _words(row.get("transcript"))
        if not ref:
            continue
        dist, miss = _align(ref, hyp)
        errors += dist
        words += len(ref)
        count += 1
        missed.update(miss)
    return ArchiveScore(errors / words if words else 0.0, count, missed.most_common(top), skipped)
