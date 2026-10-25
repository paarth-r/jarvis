"""Tests for the voice permission policy and spoken-confirmation parsing."""
from __future__ import annotations

import pytest

from jarvis.voice.permissions import classify_tool, describe_action, is_affirmative

SELF = "iMessage;-;+15550001111"


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "Mobile Documents" / "vault"
    (v / "wiki").mkdir(parents=True)
    return v


def test_classify(vault, tmp_path):
    inside = str(vault / "wiki" / "a.md")
    cases = [
        ("Bash", {"command": "ls"}, "confirm"),
        ("Edit", {"file_path": inside}, "allow"),
        ("Write", {"file_path": inside}, "allow"),
        ("NotebookEdit", {"notebook_path": inside}, "allow"),
        ("NotebookEdit", {"notebook_path": "/Users/x/n.ipynb"}, "confirm"),
        ("Edit", {"file_path": "/Users/x/Code/a.py"}, "confirm"),
        ("Write", {"file_path": "relative.md"}, "confirm"),
        ("Write", {"file_path": str(vault / ".." / "evil.md")}, "confirm"),
        ("Write", {}, "confirm"),
        ("Read", {"file_path": "/etc/hosts"}, "allow"),
        ("Grep", {"pattern": "x"}, "allow"),
        ("Glob", {"pattern": "*"}, "allow"),
        ("mcp__claude_ai_Gmail__send_message", {}, "confirm"),
        ("mcp__claude_ai_Gmail__create_draft", {}, "allow"),
        ("mcp__claude_ai_Gmail__forward", {}, "confirm"),
        ("mcp__claude_ai_Google_Calendar__delete_event", {}, "confirm"),
        ("mcp__claude_ai_Google_Calendar__respond_to_event", {}, "confirm"),
        ("mcp__claude_ai_Google_Drive__share_file", {}, "confirm"),
        ("mcp__claude_ai_Google_Drive__trash_file", {}, "confirm"),
        ("mcp__notion__notion-calendar-cancel-events", {}, "confirm"),
        ("mcp__claude_ai_Gmail__search_threads", {}, "allow"),
        ("mcp__room__set_state", {}, "allow"),
        ("mcp__plugin_imessage_imessage__reply", {"chat_id": SELF}, "allow"),
        ("mcp__plugin_imessage_imessage__reply", {"chat_id": "other"}, "confirm"),
    ]
    for name, inp, want in cases:
        assert classify_tool(name, inp, vault, SELF) == want, (name, inp)


def test_imessage_reply_without_self_chat_confirms(vault):
    name = "mcp__plugin_imessage_imessage__reply"
    assert classify_tool(name, {"chat_id": None}, vault, None) == "confirm"
    assert classify_tool(name, {}, vault) == "confirm"


def test_affirmative():
    yes = ["Yes.", "yeah", "YEP!", "yes   please", "send it", "Go ahead.", "do it", "confirm", "Yes, send it!"]
    no = ["yes no wait", "", None, "no", "yes but not that one", "   ", "maybe"]
    for t in yes:
        assert is_affirmative(t), t
    for t in no:
        assert not is_affirmative(t), t


def test_describe_gmail():
    out = describe_action(
        "mcp__claude_ai_Gmail__send_message",
        {"to": "bob@example.com", "subject": "Lunch"},
    )
    assert out == "Send an email to bob@example.com with subject Lunch?"


def test_describe_bash_truncates_and_is_speakable():
    out = describe_action("Bash", {"command": "echo " + "x" * 300})
    assert out.startswith("Run the shell command: echo ")
    assert len(out) < 140
    assert "`" not in out and "*" not in out


def test_describe_fallbacks():
    assert describe_action("mcp__claude_ai_Google_Drive__share_file", {}) == "Use share file?"
    assert describe_action("Write", {"file_path": "/a/b.txt"}) == "Write the file b.txt?"
    assert "delete" in describe_action("mcp__claude_ai_Google_Calendar__delete_event", {"event_id": "1"}).lower()
