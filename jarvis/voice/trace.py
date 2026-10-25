"""Readable trace of what the assistant hears, thinks, does and says, on stderr.

Lines look like ``[heard] what's due tomorrow`` or ``[tool] Bash: list the vault``,
next to the ``[voice]`` state transitions.
"""
from __future__ import annotations

import json
import sys

MAX_CHARS = 400


def trace(kind: str, text: object, limit: int = MAX_CHARS) -> None:
    line = " ".join(str(text).split())
    if not line:
        return
    if len(line) > limit:
        line = line[:limit].rstrip() + "..."
    print(f"[{kind}] {line}", file=sys.stderr, flush=True)


def tool_summary(name: str, tool_input: dict) -> str:
    """One line per tool call: Claude's description when there is one, else the input."""
    desc = tool_input.get("description")
    if desc:
        return f"{name}: {desc}"
    try:
        args = json.dumps(tool_input, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        args = str(tool_input)
    return f"{name} {args}"
