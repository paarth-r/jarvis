"""Tests for the stderr trace lines."""
from __future__ import annotations

from jarvis.voice.trace import tool_summary, trace


def test_trace_is_one_clipped_line(capsys):
    trace("thinking", "first line\n  second line", limit=15)
    assert capsys.readouterr().err == "[thinking] first line seco...\n"


def test_trace_skips_empty(capsys):
    trace("said", "   ")
    assert capsys.readouterr().err == ""


def test_tool_summary_prefers_description():
    assert tool_summary("Bash", {"command": "ls -la", "description": "List files"}) == "Bash: List files"
    assert tool_summary("Read", {"file_path": "/a.md"}) == 'Read {"file_path": "/a.md"}'
