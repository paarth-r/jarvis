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


def test_round2_rules(vault):
    cal = "mcp__claude_ai_Google_Calendar__"
    cases = [
        (cal + "create_event", {"attendeeEmails": ["a@b.c"]}, "confirm"),
        (cal + "update_event", {"addedAttendees": [{"email": "a@b.c"}]}, "confirm"),
        (cal + "update_event", {"summary": "tennis"}, "confirm"),
        ("mcp__notion__notion-calendar-update-events", {}, "confirm"),
        ("mcp__claude_ai_Gmail__create_draft_and_post", {}, "confirm"),
        (cal + "create_event_and_invite", {}, "confirm"),
        ("mcp__claude_ai_Gmail__create_draft", {"to": ["a@x"]}, "allow"),
        ("mcp__notion__notion-search", {}, "allow"),
        (cal + "update_event", {"addedAttendeeEmails": "a@b.c"}, "confirm"),
        (cal + "create_event", {"summary": "tennis"}, "allow"),
        (cal + "list_events", {}, "allow"),
        ("mcp__claude-in-chrome__computer", {}, "confirm"),
        ("mcp__claude-in-chrome__javascript_tool", {}, "confirm"),
        ("mcp__claude-in-chrome__form_input", {}, "confirm"),
        ("mcp__claude-in-chrome__file_upload", {}, "confirm"),
        ("mcp__notion__notion-update-page", {}, "confirm"),
        ("mcp__notion__notion-move-pages", {}, "confirm"),
        ("mcp__claude_ai_Google_Drive__update_file", {}, "confirm"),
        ("mcp__claude_ai_Google_Drive__create_file", {}, "confirm"),
        ("mcp__claude_ai_Gmail__search_threads", {}, "allow"),
        ("mcp__claude_ai_Google_Drive__download_file_content", {}, "allow"),
        ("mcp__notion__notion-fetch", {}, "allow"),
        ("mcp__notion__notion-list-shared-pages", {}, "confirm"),
        ("mcp__room__anything", {}, "allow"),
    ]
    for probe in (
        "update_file_status", "forget_x", "target_update", "widget_create",
        "overwrite_budget", "unlist_page", "wipe_list", "checkout_and_push",
        "export_and_publish", "readd_member", "finder_execute",
        "queryset_drop", "set_status", "mark_read",
    ):
        assert classify_tool("mcp__srv__" + probe, {}, vault) == "confirm", probe
    for name, inp, want in cases:
        assert classify_tool(name, inp, vault) == want, (name, inp)
    for sub in (".GIT/config", ".Claude/x", "CLAUDE.md", "claude.local.md", ".MCP.json"):
        assert classify_tool("Write", {"file_path": str(vault / sub)}, vault) == "confirm", sub
    assert classify_tool("Write", {"file_path": str(vault / "wiki" / "CLAUDE.md")}, vault) == "allow"


def test_round2_readbacks():
    g = describe_action(
        "mcp__claude_ai_Gmail__send_message",
        {"to": ["a@x.com"], "cc": ["c@x.com"], "bcc": ["b@x.com"], "subject": "Hi", "replyAll": True},
    )
    assert "a@x.com" in g and "c@x.com" in g and "and blind copy to b@x.com" in g
    assert "reply all" in g.lower()
    d = describe_action("mcp__claude_ai_Gmail__send_message", {"draftId": "d1", "subject": "Plans"})
    assert d == "Send your saved draft?"
    assert describe_action(
        "mcp__claude_ai_Gmail__send_message", {"draftId": "d", "to": ["a@x"]}
    ) == "Send your saved draft?"
    assert describe_action("mcp__claude_ai_Gmail__reply", {"replyAll": True}) == "Reply all to the email thread?"
    assert describe_action("mcp__claude_ai_Gmail__reply", {}) == "Reply to the email thread?"
    assert describe_action("mcp__claude_ai_Gmail__forward", {"to": ["a@x"]}) == "Forward the email to a@x?"
    c = describe_action(
        "mcp__claude_ai_Google_Calendar__create_event",
        {"summary": "T", "attendeeEmails": "a@b.c", "guestPermissions": {"x": 1}},
    )
    assert c.endswith("with 1 attendee?")
    assert describe_action("mcp__claude_ai_Gmail__send_message", {"draftId": "d1"}) == "Send your saved draft?"
    b = describe_action("Bash", {"command": "echo a\nrm -rf x", "description": "Remove the x folder"})
    assert b == "I want to remove the x folder. Approve?"
    s = describe_action(
        "mcp__claude_ai_Google_Drive__share_file",
        {"fileId": "f1", "emailAddress": "z@x.com", "chatSpaceId": "sp", "role": "writer"},
    )
    assert "z@x.com" in s and "a chat space" in s and "writer" in s


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
    # A long command with a description is approved from the description.
    assert not too_long_to_confirm("Bash", {"command": "x" * 500, "description": "Clean the cache"})


def test_describe_targets():
    g = describe_action("mcp__claude_ai_Gmail__send_message", {"to": ["a@x.com", "b@x.com"], "subject": "Hi"})
    assert "a@x.com" in g and "b@x.com" in g
    m = describe_action("mcp__plugin_imessage_imessage__reply", {"chat_id": "chat42", "text": "hello"})
    assert "chat42" in m and "hello" in m
    e = describe_action("mcp__claude_ai_Google_Calendar__create_event", {"summary": "Standup", "attendees": ["a", "b"]})
    assert "Standup" in e and "2 attendees" in e
    s = describe_action("mcp__claude_ai_Google_Drive__share_file", {"fileId": "f1", "emailAddress": "z@x.com"})
    assert "z@x.com" in s
    assert len(describe_action("mcp__claude_ai_Gmail__send_message", {"to": "a", "subject": "s" * 300})) < 200


def test_imessage_reply_without_self_chat_confirms(vault):
    name = "mcp__plugin_imessage_imessage__reply"
    assert classify_tool(name, {"chat_id": None}, vault, None) == "confirm"
    assert classify_tool(name, {}, vault) == "confirm"


def test_affirmative():
    yes = ["Yes.", "yeah", "YEP!", "yes   please", "send it", "Go ahead.", "do it", "confirm", "Yes, send it!", "Approve.", "approved", "Okay.", "sure"]
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


def test_describe_bash_speaks_the_request_not_the_command():
    cmd = "cat ~/a* | grep x > o.txt && rm -rf ~/old_logs"
    out = describe_action("Bash", {"command": cmd, "description": "Delete the old logs folder."})
    assert out == "I want to delete the old logs folder. Approve?"
    assert describe_action("Bash", {"command": cmd}) == "I want to run a shell command. Approve?"
    for part in ("rm", "grep", "~", "|", "old_logs"):
        assert part not in describe_action("Bash", {"command": cmd})


def test_describe_fallbacks():
    assert describe_action("mcp__claude_ai_Google_Drive__share_file", {}) == "Share a file with someone?"
    assert describe_action("Write", {"file_path": "/a/b.txt"}) == "Write the file b.txt?"
    assert "delete" in describe_action("mcp__claude_ai_Google_Calendar__delete_event", {"event_id": "1"}).lower()


def test_round4_reply_forward_readbacks():
    reply = "mcp__claude_ai_Gmail__reply"
    fwd = "mcp__claude_ai_Gmail__forward"
    assert describe_action(
        reply, {"messageId": "m", "body": "b", "to": ["x@a.com"], "cc": ["y@a.com"], "bcc": ["z@a.com"]}
    ) == "Reply to the thread, sending to x@a.com, copying y@a.com, and blind copying z@a.com?"
    assert describe_action(reply, {"messageId": "m", "bcc": ["z@a.com"]}) == (
        "Reply to the thread, blind copying z@a.com?"
    )
    assert describe_action(reply, {"messageId": "m", "replyAll": True, "cc": ["y@a.com"]}) == (
        "Reply all to the thread, copying y@a.com?"
    )
    assert describe_action(reply, {"messageId": "m", "replyAll": True}) == "Reply all to the email thread?"
    assert describe_action(reply, {"messageId": "m", "to": [], "cc": []}) == "Reply to the email thread?"
    assert describe_action(
        fwd, {"messageId": "m", "to": ["x@a.com"], "cc": ["y@a.com"], "bcc": ["z@a.com"]}
    ) == "Forward the email to x@a.com, copying y@a.com, and blind copying z@a.com?"
    assert describe_action(fwd, {"messageId": "m", "bcc": ["z@a.com"]}) == (
        "Forward the email to someone, blind copying z@a.com?"
    )
    many = [f"person{i}@example.com" for i in range(8)]
    out = describe_action(reply, {"messageId": "m", "to": many})
    assert all(p in out for p in many) and "..." not in out


def test_round4_mcp_name_rules(vault):
    confirm = [
        "mcp__run__list_run", "mcp__a_push__list_push", "mcp__push__push_list_push",
        "mcp__notion__notion-notion-list-pages", "mcp__srv__srv_srv_list",
        "mcp__srv__list_and_get", "mcp__srv__get_or_list", "mcp__srv__check_out",
        "mcp__srv__list_and", "mcp____list", "mcp__srv__", "mcp__",
        "mcp__srv__get_approve", "mcp__srv__list_pay", "mcp__srv__get_archive",
        "mcp__srv__get_rename", "mcp__srv__get_reset", "mcp__srv__get_restore",
        "mcp__srv__get_label", "mcp__srv__get_copy", "mcp__srv__get_duplicate",
        "mcp__srv__get_convert", "mcp__srv__get_stop", "mcp__srv__get_resolve",
        "mcp__srv__get_react", "mcp__srv__get_purchase", "mcp__srv__get_unlabel",
        "mcp__srv__get_untrash", "mcp__srv__get_apply", "mcp__srv__get_mark",
        "mcp__claude_ai_Gmail__untrash_thread", "mcp__claude_ai_Gmail__label_thread",
        "mcp__claude_ai_Gmail__apply_sensitive_message_label",
        "mcp__claude_ai_Google_Drive__copy_file", "mcp__notion__notion-duplicate-page",
        "mcp__notion__notion-restore-pages", "mcp__notion__notion-stop-session",
        "mcp__notion__notion-set-comment-reaction", "mcp__notion__notion-convert-page-to-skill",
    ]
    for name in confirm:
        assert classify_tool(name, {}, vault) == "confirm", name
    allow = [
        "mcp__claude_ai_Gmail__search_threads", "mcp__claude_ai_Gmail__get_thread",
        "mcp__claude_ai_Gmail__create_draft", "mcp__claude_ai_Google_Calendar__list_events",
        "mcp__claude_ai_Google_Drive__read_file_content",
        "mcp__claude_ai_Google_Drive__download_file_content",
        "mcp__notion__notion-fetch", "mcp__notion__notion-search",
        "mcp__notion__notion-calendar-list-events", "mcp__room__set_state",
    ]
    for name in allow:
        assert classify_tool(name, {}, vault) == "allow", name
    cal = "mcp__claude_ai_Google_Calendar__"
    assert classify_tool(cal + "create_event", {"summary": "tennis"}, vault) == "allow"
    assert classify_tool(cal + "create_event", {"attendeeEmails": ["a@b.c"]}, vault) == "confirm"
    assert classify_tool(cal + "update_event", {}, vault) == "confirm"


def test_round4_calendar_readbacks():
    cal = "mcp__claude_ai_Google_Calendar__create_event"
    out = describe_action(cal, {"summary": "Tennis", "attendees": {"a": {"email": "a@b.c"}}})
    assert out.endswith("Tennis with attendees?")
    n = "mcp__notion__notion-calendar-update-events"
    assert describe_action(n, {}) == "Change a Notion calendar event?"
    assert describe_action(n, {"events": [{"attendees": {"x": 1}}]}) == "Change a Notion calendar event with attendees?"
    assert describe_action("mcp__notion__notion-calendar-create-events", {"events": [{"title": "Lunch"}]}) == (
        "Create the Notion calendar event Lunch?"
    )
    assert describe_action("mcp__notion__notion-calendar-cancel-events", {}) == "Cancel a Notion calendar event?"
    assert describe_action(
        "mcp__notion__notion-calendar-create-events", {"events": [{"title": "L", "attendees": ["a", "b"]}]}
    ) == "Create the Notion calendar event L with 2 attendees?"


def _cal(summary, attendees=None):
    item = {"calendar": {"accountId": "a", "calendarId": "c"}, "eventId": "e"}
    if summary:
        item["eventSummary"] = summary
    return item


def test_round5_notion_multi_event_readbacks():
    cancel = "mcp__notion__notion-calendar-cancel-events"
    out = describe_action(cancel, {"events": [_cal("Standup"), _cal("Board meeting"), _cal("Wedding")]})
    assert out == "Cancel 3 Notion calendar events: Standup, Board meeting, and Wedding?"
    assert describe_action(cancel, {"events": [_cal("Standup")]}) == "Cancel the Notion calendar event Standup?"
    assert describe_action(cancel, {"events": [_cal("A"), _cal(None)]}) == "Cancel 2 Notion calendar events?"
    create = "mcp__notion__notion-calendar-create-events"
    ev = [
        {"summary": "Lunch", "attendees": [{"email": "a@x.com"}]},
        {"summary": "Tennis", "attendees": [{"email": "b@x.com"}, {"email": "c@x.com"}]},
        {"summary": "Gym"},
    ]
    assert describe_action(create, {"timeZone": "UTC", "events": ev}) == (
        "Create 3 Notion calendar events: Lunch with 1 attendee, Tennis with 2 attendees, and Gym?"
    )
    upd = "mcp__notion__notion-calendar-update-events"
    updates = [
        {"updateType": "UPDATE", "event": _cal(None), "update": {"attendees": [{"email": "a@x.com"}]}},
        {"updateType": "UPDATE", "event": _cal(None), "update": {"location": "x"}},
    ]
    assert describe_action(upd, {"timeZone": "UTC", "updates": updates}) == (
        "Change 2 Notion calendar events, some with attendees?"
    )
    two = [{"updateType": "UPDATE", "event": _cal("A"), "update": {}},
           {"updateType": "UPDATE", "event": _cal("B"), "update": {"summary": "B2"}}]
    assert describe_action(upd, {"timeZone": "UTC", "updates": two}) == "Change 2 Notion calendar events: A and B?"


def test_round5_recipient_keys_and_formatting():
    send = "mcp__claude_ai_Gmail__send_message"
    out = describe_action(send, {"To": ["a@x.com"], "CC": ["c@x.com"], "BCC": ["b@x.com"], "subject": "s"})
    assert "a@x.com" in out and "copying c@x.com" in out and "blind copy to b@x.com" in out
    out = describe_action(send, {"message": {"to": ["n@x.com"], "Bcc": "h@x.com"}, "subject": "s"})
    assert "n@x.com" in out and "h@x.com" in out
    reply = "mcp__claude_ai_Gmail__reply"
    assert describe_action(reply, {"messageId": "m", "to": ["  "], "cc": [""], "Bcc": " "}) == (
        "Reply to the email thread?"
    )
    assert describe_action(reply, {"messageId": "m", "to": ["john_doe@x.com"]}) == (
        "Reply to the thread, sending to john_doe@x.com?"
    )
    assert describe_action("mcp__claude_ai_Gmail__forward", {"messageId": "m", "To": ["a_b@x.com"]}) == (
        "Forward the email to a_b@x.com?"
    )
    one = describe_action("mcp__claude_ai_Google_Calendar__create_event", {"summary": "T", "attendees": ["a"]})
    assert one.endswith("with 1 attendee?")


def test_round5_long_readback_is_too_long():
    many = [f"person{i}@example.com" for i in range(200)]
    for name, inp in (
        ("mcp__claude_ai_Gmail__send_message", {"to": many, "subject": "s"}),
        ("mcp__claude_ai_Gmail__reply", {"messageId": "m", "bcc": many}),
        ("mcp__claude_ai_Gmail__forward", {"messageId": "m", "cc": many}),
    ):
        assert too_long_to_confirm(name, inp), name
    assert not too_long_to_confirm("mcp__claude_ai_Gmail__send_message", {"to": ["a@x.com"], "subject": "s"})
    assert too_long_to_confirm("Bash", {"command": "x" * 201})
    assert not too_long_to_confirm("Bash", {"command": "ls"})


def test_dict_recipient_without_known_keys_is_still_read_back():
    send = "mcp__claude_ai_Gmail__send_message"
    out = describe_action(send, {"to": [{"handle": "mystery-recipient"}], "subject": "s"})
    assert "mystery-recipient" in out


def test_whitespace_only_titles_count_as_missing():
    create = "mcp__notion__notion-calendar-create-events"
    assert describe_action(create, {"events": [{"title": "   "}]}) == (
        "Create a Notion calendar event?"
    )
    cancel = "mcp__notion__notion-calendar-cancel-events"
    assert describe_action(cancel, {"events": [_cal("A"), _cal(" ")]}) == (
        "Cancel 2 Notion calendar events?"
    )
    cal = "mcp__claude_ai_Google_Calendar__create_event"
    assert describe_action(cal, {"summary": " \n "}) == "Create event an event?"


@pytest.mark.parametrize(
    ("name", "tool_input", "expected"),
    [
        ("Bash", {"command": "rm -rf build"}, "allow"),
        ("Write", {"file_path": "/etc/hosts"}, "allow"),
        ("Agent", {}, "allow"),
        ("mcp__claude-in-chrome__computer", {}, "allow"),
        ("mcp__notion__notion-update-page", {}, "allow"),
        ("mcp__claude_ai_Gmail__send_message", {"to": ["a@x.com"]}, "confirm"),
        ("mcp__claude_ai_Gmail__reply", {}, "confirm"),
        ("mcp__claude_ai_Google_Drive__share_file", {}, "confirm"),
        ("mcp__claude_ai_Google_Drive__trash_file", {}, "confirm"),
        ("mcp__claude_ai_Google_Calendar__create_event", {"summary": "Tennis"}, "allow"),
        ("mcp__claude_ai_Google_Calendar__create_event", {"attendeeEmails": ["a@x.com"]}, "confirm"),
        ("mcp__claude_ai_Google_Calendar__update_event", {"summary": "Tennis"}, "confirm"),
        ("mcp__plugin_imessage_imessage__reply", {"chat_id": "other"}, "confirm"),
        ("mcp__plugin_imessage_imessage__reply", {"chat_id": "me"}, "allow"),
        ("mcp__room__set_state", {}, "allow"),
    ],
)
def test_auto_mode_only_gates_outward_actions(tmp_path, name, tool_input, expected):
    assert classify_tool(name, tool_input, tmp_path, "me", auto=True) == expected
