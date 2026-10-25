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
CONFIRM_MCP_SUBSTRINGS = (
    "send", "reply", "forward", "share", "delete", "trash",
    "respond_to_event", "cancel", "spawn", "comment",
)
CALENDAR_WRITES = ("create_event", "update_event")
ATTENDEE_KEYS = {"attendees", "guests"}
AFFIRMATIVES = {
    "yes", "yeah", "yep", "yes please", "do it", "send it", "go ahead",
    "confirm", "yes send it", "yes do it",
}
MAX_SPOKEN = 80
MAX_BASH_CONFIRM = 200


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
    return not (set(target.relative_to(root).parts) & PROTECTED_DIRS)


def _has_attendees(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ATTENDEE_KEYS and isinstance(item, (list, tuple)) and item:
                return True
            if _has_attendees(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_has_attendees(i) for i in value)
    return False


def classify_tool(
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
    if any(s in norm for s in CALENDAR_WRITES) and _has_attendees(tool_input):
        return "confirm"
    return "allow"


def too_long_to_confirm(name: str, tool_input: dict) -> bool:
    """True when the action cannot be read back in full (Bash over 200 chars)."""
    return name == "Bash" and len(str(tool_input.get("command", ""))) > MAX_BASH_CONFIRM


def is_affirmative(transcript: str | None) -> bool:
    if not transcript:
        return False
    text = re.sub(r"[^\w\s]", "", transcript.lower())
    return " ".join(text.split()) in AFFIRMATIVES


_SYMBOL_WORDS = {
    "&&": " and then ", "||": " or ", ">>": " append to ", "|": " pipe ",
    ">": " redirect to ", "<": " read from ", ";": " then ", "*": " star ",
    "~": " home ", "`": " backtick ", "$": " dollar ", "&": " ampersand ",
    "#": " hash ", "\\": " backslash ",
}
_SYMBOL_RE = re.compile("|".join(re.escape(k) for k in _SYMBOL_WORDS))


def _speakable_command(command: str) -> str:
    text = _SYMBOL_RE.sub(lambda m: _SYMBOL_WORDS[m.group(0)], command)
    return " ".join(text.split())


def _clip(value: object, limit: int = MAX_SPOKEN) -> str:
    text = " ".join(str(value).split())
    for ch in "`*#_[]<>|":
        text = text.replace(ch, "")
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _people(value: object) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    return _clip(value) if value else "someone"


def _short_name(name: str) -> str:
    base = name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name
    return base.replace("_", " ").replace("-", " ")


def _count_attendees(value: object) -> int:
    n = 0
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ATTENDEE_KEYS and isinstance(item, (list, tuple)):
                n += len(item)
            else:
                n += _count_attendees(item)
    elif isinstance(value, (list, tuple)):
        n = sum(_count_attendees(i) for i in value)
    return n


def _find_key(value: object, keys: tuple[str, ...]) -> object:
    if isinstance(value, dict):
        for k in keys:
            if value.get(k):
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


def describe_action(name: str, tool_input: dict) -> str:
    if name == "Bash":
        return f"Run the shell command: {_speakable_command(str(tool_input.get('command', '')))}?"
    if name in PATH_CHECKED_WRITERS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        verb = "Write" if name == "Write" else "Edit"
        return f"{verb} the file {Path(str(path)).name or 'unknown'}?"
    norm = name.lower().replace("-", "_")
    if norm.startswith("mcp__"):
        if "gmail" in norm and ("send" in norm or "forward" in norm or "reply" in norm):
            to = _people(tool_input.get("to"))
            subject = _clip(tool_input.get("subject", "no subject"))
            return f"Send an email to {to} with subject {subject}?"
        if "imessage" in norm and norm.endswith("reply"):
            target = tool_input.get("chat_id") or tool_input.get("recipient") or "unknown chat"
            return f"Send a text message to {_clip(target)}: {_clip(tool_input.get('text', ''))}?"
        if "share" in norm:
            target = _find_key(tool_input, ("email", "emailAddress", "email_address", "to", "user", "target"))
            file_id = tool_input.get("file_id") or tool_input.get("fileId") or "a file"
            return f"Share {_clip(file_id)} with {_people(target)}?"
        if any(s in norm for s in CALENDAR_WRITES) or "respond_to_event" in norm:
            title = _find_key(tool_input, ("summary", "title")) or "an event"
            count = _count_attendees(tool_input)
            tail = f" with {count} attendees" if count else ""
            return f"{_short_name(name).capitalize()} {_clip(title)}{tail}?"
    return f"Use {_short_name(name)}?"
