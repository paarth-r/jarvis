"""Voice permission policy: which tool calls need a spoken yes, and how to ask."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

CONFIRM_MCP_SUBSTRINGS = (
    "send_message",
    "reply",
    "forward",
    "share_file",
    "delete",
    "trash",
    "respond_to_event",
    "cancel",
)
FILE_WRITE_TOOLS = {"Write", "Edit", "NotebookEdit"}
AFFIRMATIVES = {
    "yes", "yeah", "yep", "yes please", "do it", "send it", "go ahead",
    "confirm", "yes send it", "yes do it",
}
MAX_SPOKEN = 80


def _inside(path_str: object, vault: Path) -> bool:
    if not isinstance(path_str, str) or not path_str:
        return False
    try:
        target = Path(path_str).expanduser().resolve()
        root = Path(vault).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    return target.is_relative_to(root)


def classify_tool(
    name: str,
    tool_input: dict,
    vault: Path,
    self_chat_id: str | None = None,
) -> Literal["allow", "confirm"]:
    if name == "Bash":
        return "confirm"
    if name in FILE_WRITE_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path")
        return "allow" if _inside(path, vault) else "confirm"
    if name.startswith("mcp__"):
        lowered = name.lower()
        if not any(s in lowered for s in CONFIRM_MCP_SUBSTRINGS):
            return "allow"
        if (
            "imessage" in lowered
            and lowered.endswith("reply")
            and self_chat_id is not None
            and tool_input.get("chat_id") == self_chat_id
        ):
            return "allow"
        return "confirm"
    return "allow"


def is_affirmative(transcript: str | None) -> bool:
    if not transcript:
        return False
    text = re.sub(r"[^\w\s]", "", transcript.lower())
    return " ".join(text.split()) in AFFIRMATIVES


def _clip(value: object, limit: int = MAX_SPOKEN) -> str:
    text = " ".join(str(value).split())
    for ch in "`*#_[]<>|":
        text = text.replace(ch, "")
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _short_name(name: str) -> str:
    base = name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name
    return base.replace("_", " ").replace("-", " ")


def describe_action(name: str, tool_input: dict) -> str:
    if name == "Bash":
        return f"Run the shell command: {_clip(tool_input.get('command', ''))}?"
    if name in FILE_WRITE_TOOLS:
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        verb = "Edit" if name != "Write" else "Write"
        return f"{verb} the file {Path(str(path)).name or 'unknown'}?"
    lowered = name.lower()
    if lowered.startswith("mcp__") and lowered.endswith("send_message") and "gmail" in lowered:
        to = _clip(tool_input.get("to", "someone"))
        subject = _clip(tool_input.get("subject", "no subject"))
        return f"Send an email to {to} with subject {subject}?"
    if lowered.endswith("imessage__reply"):
        return f"Send a text message: {_clip(tool_input.get('text', ''))}?"
    return f"Use {_short_name(name)}?"
