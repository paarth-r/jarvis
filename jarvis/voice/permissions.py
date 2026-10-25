"""Voice permission policy: which tool calls need a spoken yes, and how to ask.

The policy fails closed: only an explicit allowlist of built-ins is allowed,
every unknown non-MCP tool needs a spoken confirmation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

SAFE_BUILTINS = {
    "Read", "Grep", "Glob", "LS", "WebSearch", "WebFetch", "TodoWrite",
    "ToolSearch", "Skill",
}
PATH_CHECKED_WRITERS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
PROTECTED_DIRS = {".claude", ".git", ".obsidian"}
PROTECTED_ROOT_FILES = {"claude.md", "claude.local.md", ".mcp.json"}
CONFIRM_MCP_SUBSTRINGS = (
    "send", "reply", "forward", "share", "delete", "trash",
    "respond_to_event", "cancel", "spawn", "comment",
)
READ_VERBS = {
    "list", "get", "search", "read", "query", "fetch", "find", "check",
    "download", "export", "status",
}
MUTATING_WORDS = (
    "send", "reply", "forward", "share", "delete", "trash", "cancel", "spawn",
    "comment", "update", "create", "write", "set", "mark", "move", "remove",
    "drop", "push", "publish", "execute", "run", "post", "add", "edit",
    "upload", "invite", "respond", "approve", "pay", "archive", "rename",
    "reset", "restore", "label", "copy", "duplicate", "convert", "stop",
    "resolve", "react", "purchase", "out", "unlabel", "untrash", "apply",
)
CONJUNCTIONS = {"and", "or"}
AFFIRMATIVES = {
    "yes", "yeah", "yep", "yes please", "do it", "send it", "go ahead",
    "confirm", "yes send it", "yes do it", "approve", "approved", "i approve",
    "yes approve", "okay", "ok", "sure", "go for it",
}
MAX_SPOKEN = 80
MAX_BASH_CONFIRM = 200
MAX_READBACK = 300


def _inside(path_str: object, vault: Path) -> bool:
    """True if path resolves inside the vault and not under a protected dir."""
    if not isinstance(path_str, str) or not path_str:
        return False
    try:
        target = Path(path_str).expanduser().resolve()
        root = Path(vault).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    if not target.is_relative_to(root):
        return False
    parts = [part.lower() for part in target.relative_to(root).parts]
    if set(parts) & PROTECTED_DIRS:
        return False
    return not (parts and parts[0] in PROTECTED_ROOT_FILES)


def _is_attendee_key(key: object) -> bool:
    lowered = str(key).lower()
    return "attendee" in lowered or "guest" in lowered


def _has_attendees(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_attendee_key(key) and item:
                return True
            if _has_attendees(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_attendees(i) for i in value)
    return False


def _tool_part(norm: str) -> str:
    """Part of a normalized MCP name after the server prefix."""
    return norm.split("__", 2)[-1]


# In auto mode Claude Code's own classifier vets every tool call, so the voice
# gate keeps only actions that reach other people or can't be undone.
OUTWARD_WORDS = (
    "send", "reply", "forward", "share", "invite", "respond", "delete", "trash",
    "comment", "spawn", "post", "publish", "upload",
)


def classify_tool(
    name: str,
    tool_input: dict,
    vault: Path,
    self_chat_id: str | None = None,
    auto: bool = False,
) -> Literal["allow", "confirm"]:
    verdict = _classify_strict(name, tool_input, vault, self_chat_id)
    if not auto or verdict == "allow" or not name.startswith("mcp__"):
        return "allow" if auto else verdict
    norm = name.lower().replace("-", "_")
    tokens = [t for t in norm.split("__", 2)[-1].split("_") if t]
    if any(t.startswith(w) for t in tokens for w in OUTWARD_WORDS):
        return "confirm"
    if "update_event" in norm or _has_attendees(tool_input):
        return "confirm"
    return "allow"


def _classify_strict(
    name: str,
    tool_input: dict,
    vault: Path,
    self_chat_id: str | None = None,
) -> Literal["allow", "confirm"]:
    if name in SAFE_BUILTINS:
        return "allow"
    if name in PATH_CHECKED_WRITERS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path")
        return "allow" if _inside(path, vault) else "confirm"
    if not name.startswith("mcp__"):
        return "confirm"
    norm = name.lower().replace("-", "_")
    if any(s in norm for s in CONFIRM_MCP_SUBSTRINGS):
        if (
            "imessage" in norm
            and norm.endswith("reply")
            and self_chat_id is not None
            and tool_input.get("chat_id") == self_chat_id
        ):
            return "allow"
        return "confirm"
    if norm.startswith("mcp__room__"):
        return "allow"
    segments = norm.split("__", 2)
    if len(segments) < 3 or not segments[1].strip("_") or not segments[2].strip("_"):
        return "confirm"
    server_last = segments[1].split("_")[-1]
    body = [t for t in segments[2].split("_") if t]
    # Strip at most one leading server-echo token, then at most one "calendar"
    # token (notion-calendar-list-events -> list_events). Nothing else is skipped.
    if len(body) > 1 and body[0] == server_last:
        body = body[1:]
    if len(body) > 1 and body[0] == "calendar":
        body = body[1:]
    if CONJUNCTIONS & set(body):
        return "confirm"
    whole = "_".join(body)
    if whole == "create_draft":
        return "allow"
    if whole in {"create_event", "create_events"}:
        return "confirm" if _has_attendees(tool_input) else "allow"
    if body and body[0] in READ_VERBS and not any(
        t.startswith(w) for t in body for w in MUTATING_WORDS
    ):
        return "allow"
    return "confirm"


def too_long_to_confirm(name: str, tool_input: dict) -> bool:
    """True when the action cannot be read back in full.

    Bash commands over 200 chars with no description to speak, or any spoken
    read-back over 300 chars (e.g. a 200-address recipient list), are too long
    to approve by voice.
    """
    if (
        name == "Bash"
        and not _bash_description(tool_input)
        and len(str(tool_input.get("command", ""))) > MAX_BASH_CONFIRM
    ):
        return True
    try:
        return len(describe_action(name, tool_input)) > MAX_READBACK
    except Exception:
        return True


def is_affirmative(transcript: str | None) -> bool:
    if not transcript:
        return False
    text = re.sub(r"[^\w\s]", "", transcript.lower())
    return " ".join(text.split()) in AFFIRMATIVES


def _bash_description(tool_input: dict) -> str:
    """Claude's plain-English summary of a shell command, ready to speak.

    The command itself is never read aloud; the request is phrased from this.
    """
    text = _clip(tool_input.get("description") or "", MAX_SPOKEN).strip().rstrip(".")
    return text[:1].lower() + text[1:] if text else ""


def _clip(value: object, limit: int = MAX_SPOKEN) -> str:
    text = " ".join(str(value).split())
    for ch in "`*#_[]<>|":
        text = text.replace(ch, "")
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _person(value: object) -> str:
    if isinstance(value, dict):
        # Unknown shapes fall back to the whole dict: a recipient is never dropped.
        value = value.get("email") or value.get("address") or value.get("name") or str(value)
    text = " ".join(str(value).split())
    for ch in "`*#[]<>|":  # keep "_" so john_doe@x.com stays intact
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _people(value: object) -> str:
    """Every recipient, never truncated: hiding who receives something is unsafe."""
    items = value if isinstance(value, (list, tuple)) else [value]
    names = [p for p in (_person(v) for v in items if v) if p]
    return ", ".join(names) if names else "someone"


def _recipients(tool_input: dict, key: str) -> list:
    """Non-blank recipients under `key` (any case), top level or one dict level down."""
    found: list = []

    def take(d: dict) -> None:
        for k, v in d.items():
            if str(k).lower() == key and v:
                for item in v if isinstance(v, (list, tuple)) else [v]:
                    if _person(item):
                        found.append(item)

    take(tool_input)
    for v in tool_input.values():
        if isinstance(v, dict):
            take(v)
    return found


def _join_clauses(clauses: list[str]) -> str:
    if len(clauses) <= 1:
        return "".join(clauses)
    if len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    return ", ".join(clauses[:-1]) + ", and " + clauses[-1]


def _copy_clauses(tool_input: dict) -> list[str]:
    clauses = []
    if cc := _recipients(tool_input, "cc"):
        clauses.append(f"copying {_people(cc)}")
    if bcc := _recipients(tool_input, "bcc"):
        clauses.append(f"blind copying {_people(bcc)}")
    return clauses


def _short_name(name: str) -> str:
    base = name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name
    return base.replace("_", " ").replace("-", " ")


def _count_attendees(value: object) -> int:
    n = 0
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_attendee_key(key) and item:
                if isinstance(item, (list, tuple)):
                    n += len(item)
                elif not isinstance(item, dict):
                    n += 1
            else:
                n += _count_attendees(item)
    elif isinstance(value, (list, tuple)):
        n = sum(_count_attendees(i) for i in value)
    return n


def _attendee_tail(tool_input: dict) -> str:
    count = _count_attendees(tool_input)
    if count:
        return f" with {count} attendee{'' if count == 1 else 's'}"
    return " with attendees" if _has_attendees(tool_input) else ""


def _present(value: object) -> bool:
    """Truthy and, for strings, not whitespace only."""
    return bool(value) and not (isinstance(value, str) and not value.strip())


def _find_key(value: object, keys: tuple[str, ...]) -> object:
    if isinstance(value, dict):
        for k in keys:
            if _present(value.get(k)):
                return value[k]
        for item in value.values():
            found = _find_key(item, keys)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for item in value:
            found = _find_key(item, keys)
            if found:
                return found
    return None


def _describe_gmail(name: str, tool_input: dict) -> str:
    if tool_input.get("draftId"):
        return "Send your saved draft?"
    norm = name.lower()
    if norm.endswith("reply"):
        base = "Reply all to the" if tool_input.get("replyAll") else "Reply to the"
        clauses = []
        if to := _recipients(tool_input, "to"):
            clauses.append(f"sending to {_people(to)}")
        clauses += _copy_clauses(tool_input)
        if not clauses:
            return f"{base} email thread?"
        return f"{base} thread, {_join_clauses(clauses)}?"
    if norm.endswith("forward"):
        clauses = _copy_clauses(tool_input)
        head = f"Forward the email to {_people(_recipients(tool_input, 'to'))}"
        # The "to" head is the first list item, so cc + bcc keep the serial comma.
        return f"{head}, {', and '.join(clauses)}?" if clauses else f"{head}?"
    verb = "Reply all" if tool_input.get("replyAll") else "Send an email"
    text = f"{verb} to {_people(_recipients(tool_input, 'to'))}"
    if cc := _recipients(tool_input, "cc"):
        text += f", copying {_people(cc)}"
    if bcc := _recipients(tool_input, "bcc"):
        text += f", and blind copy to {_people(bcc)}"
    return f"{text} with subject {_clip(tool_input.get('subject') or 'no subject')}?"


def _describe_share(tool_input: dict) -> str:
    file_id = tool_input.get("fileId") or tool_input.get("file_id") or "a file"
    who = tool_input.get("emailAddress")
    if tool_input.get("chatSpaceId"):
        who = "a chat space" if not who else f"{_people(who)} and a chat space"
    role = tool_input.get("role")
    tail = f" as {_clip(role)}" if role else ""
    return f"Share {_clip(file_id)} with {_people(who)}{tail}?"


_TITLE_KEYS = ("summary", "title", "eventSummary")


def _describe_notion_calendar(norm: str, tool_input: dict) -> str:
    verb = "Create" if "create" in norm else "Cancel" if "cancel" in norm else "Change"
    items = tool_input.get("events") or tool_input.get("updates")
    if isinstance(items, (list, tuple)) and len(items) > 1:
        n = len(items)
        titles = [_find_key(item, _TITLE_KEYS) for item in items]
        if all(titles):
            parts = [f"{_clip(t)}{_attendee_tail(item)}" for t, item in zip(titles, items)]
            return f"{verb} {n} Notion calendar events: {_join_clauses(parts)}?"
        some = ", some with attendees" if _has_attendees(items) else ""
        return f"{verb} {n} Notion calendar events{some}?"
    title = _find_key(tool_input, _TITLE_KEYS)
    what = f"the Notion calendar event {_clip(title)}" if title else "a Notion calendar event"
    return f"{verb} {what}{_attendee_tail(tool_input)}?"


def describe_action(name: str, tool_input: dict) -> str:
    if name == "Bash":
        desc = _bash_description(tool_input)
        return f"I want to {desc}. Approve?" if desc else "I want to run a shell command. Approve?"
    if name in PATH_CHECKED_WRITERS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        verb = "Write" if name == "Write" else "Edit"
        return f"{verb} the file {Path(str(path)).name or 'unknown'}?"
    norm = name.lower().replace("-", "_")
    if norm.startswith("mcp__"):
        if "gmail" in norm and ("send" in norm or "forward" in norm or "reply" in norm):
            return _describe_gmail(name, tool_input)
        if "imessage" in norm and norm.endswith("reply"):
            target = tool_input.get("chat_id") or tool_input.get("recipient") or "unknown chat"
            return f"Send a text message to {_clip(target)}: {_clip(tool_input.get('text', ''))}?"
        if "share" in norm:
            return _describe_share(tool_input)
        if "notion" in norm and "calendar" in norm and "event" in norm:
            return _describe_notion_calendar(norm, tool_input)
        if "create_event" in norm or "update_event" in norm or "respond_to_event" in norm:
            title = _find_key(tool_input, _TITLE_KEYS) or "an event"
            return f"{_short_name(name).capitalize()} {_clip(title)}{_attendee_tail(tool_input)}?"
    return f"Use {_short_name(name)}?"
