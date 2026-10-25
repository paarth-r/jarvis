#!/usr/bin/env python3
"""Print word error rate over corrected utterances: voice_wer.py <data_dir>."""

from __future__ import annotations

import sys
from pathlib import Path

from jarvis.voice.wer import score_archive


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    data_dir = Path(args[0]) if args else Path("~/.jarvis/voice")
    try:
        result = score_archive(data_dir)
    except FileNotFoundError:
        print(f"no utterances.jsonl in {data_dir.expanduser()}", file=sys.stderr)
        return 1
    print(f"WER: {result.wer:.3f} over {result.count} corrected utterances")
    print("Most-missed reference words (candidates for the vocab file):")
    for word, n in result.top_missed:
        print(f"  {word}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
