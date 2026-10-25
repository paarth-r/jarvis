"""Tests for the voice permission policy and spoken-confirmation parsing."""
from __future__ import annotations

import pytest

from jarvis.voice.permissions import (
    classify_tool,
    describe_action,
    is_affirmative,
    too_long_to_confirm,
)

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


def test_fail_closed_and_mcp_rules(vault):
    cases = [
        ("MultiEdit", {"file_path": "/Users/x/a.py"}, "confirm"),
        ("MultiEdit", {"file_path": str(vault / "wiki" / "a.md")}, "allow"),
        ("Agent", {}, "confirm"),
        ("Task", {}, "confirm"),
        ("KillShell", {}, "confirm"),
        ("SomeFutureTool", {}, "confirm"),
        ("LS", {}, "allow"),
        ("WebSearch", {}, "allow"),
        ("mcp__notion__notion-send-message-to-session", {}, "confirm"),
        ("mcp__notion__notion-spawn-session", {}, "confirm"),
        ("mcp__notion__notion-create-comment", {}, "confirm"),
        ("mcp__claude_ai_Gmail__send_draft", {}, "confirm"),
        ("mcp__claude_ai_Google_Calendar__create_event", {"summary": "x"}, "allow"),
        ("mcp__claude_ai_Google_Calendar__create_event", {"attendees": []}, "allow"),
        ("mcp__claude_ai_Google_Calendar__create_event", {"attendees": ["a@b.c"]}, "confirm"),
        ("mcp__claude_ai_Google_Calendar__update_event", {"event": {"guests": ["a@b.c"]}}, "confirm"),
        ("mcp__notion__notion-calendar-create-events", {"events": [{"attendees": ["a@b.c"]}]}, "confirm"),
        ("mcp__notion__notion-calendar-create-events", {"events": [{"title": "t"}]}, "allow"),
    ]
    for name, inp, want in cases:
        assert classify_tool(name, inp, vault) == want, (name, inp)


def test_path_edge_cases(vault, tmp_path, monkeypatch):
    assert classify_tool("Write", {"file_path": "a\x00b"}, vault) == "confirm"
    for sub in (".claude/hooks/x.sh", ".git/hooks/pre-commit", ".obsidian/app.json"):
        assert classify_tool("Write", {"file_path": str(vault / sub)}, vault) == "confirm", sub
    outside = tmp_path / "outside"
    outside.mkdir()
    (vault / "link").symlink_to(outside)
    assert classify_tool("Write", {"file_path": str(vault / "link" / "f.md")}, vault) == "confirm"
    home = tmp_path / "home"
    (home / "v").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    assert classify_tool("Write", {"file_path": "~/v/a.md"}, home / "v") == "allow"
    assert classify_tool("Write", {"file_path": "~/other/a.md"}, home / "v") == "confirm"


def test_too_long_and_bash_readback():
    assert not too_long_to_confirm("Bash", {"command": "x" * 200})
    assert too_long_to_confirm("Bash", {"command": "x" * 201})
    assert not too_long_to_confirm("Write", {"command": "x" * 500})
    out = describe_action("Bash", {"command": "cat ~/a* | grep x > o.txt && rm b; ls >> l"})
    for word in ("home", "star", "pipe", "redirect to", "and then", "then", "append to"):
        assert word in out
    assert not any(c in out for c in "|>&;*~")


def test_describe_targets():
    g = describe_action("mcp__claude_ai_Gmail__send_message", {"to": ["a@x.com", "b@x.com"], "subject": "Hi"})
    assert "a@x.com" in g and "b@x.com" in g
    m = describe_action("mcp__plugin_imessage_imessage__reply", {"chat_id": "chat42", "text": "hello"})
    assert "chat42" in m and "hello" in m
    e = describe_action("mcp__claude_ai_Google_Calendar__create_event", {"summary": "Standup", "attendees": ["a", "b"]})
    assert "Standup" in e and "2 attendees" in e
    s = describe_action("mcp__claude_ai_Google_Drive__share_file", {"file_id": "f1", "email": "z@x.com"})
    assert "z@x.com" in s
    assert len(describe_action("mcp__claude_ai_Gmail__send_message", {"to": "a", "subject": "s" * 300})) < 200


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
    out = describe_action("Bash", {"command": "echo " + "x" * 150})
    assert out.startswith("Run the shell command: echo ")
    assert "`" not in out and "*" not in out


def test_describe_fallbacks():
    assert describe_action("mcp__claude_ai_Google_Drive__share_file", {}) == "Share a file with someone?"
    assert describe_action("Write", {"file_path": "/a/b.txt"}) == "Write the file b.txt?"
    assert "delete" in describe_action("mcp__claude_ai_Google_Calendar__delete_event", {"event_id": "1"}).lower()
