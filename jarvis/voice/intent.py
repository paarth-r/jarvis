"""Reading approval or refusal from a spoken answer.

Obvious answers ("yes", "no") are decided instantly; anything else ("yea
approved", "eh go for it") is judged by a small Claude model. Unclear answers,
timeouts and errors all count as a refusal.
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable

from jarvis.voice.permissions import is_affirmative

JUDGE_MODEL = "claude-haiku-5-5"
JUDGE_TIMEOUT_S = 10.0
FAST_NO = {
    "no", "nope", "nah", "no thanks", "dont", "do not", "deny", "denied", "cancel",
    "stop", "wait", "nevermind", "never mind", "forget it", "not now",
}
JUDGE_PROMPT = (
    "You decide whether a spoken answer approves an action an assistant asked "
    "about. The answer comes from speech recognition, so expect slang, filler and "
    "misspellings. Reply with exactly one word: APPROVE if the person agrees, in "
    "any wording; DENY if they refuse, hesitate, change their mind or ask to wait; "
    "UNCLEAR if you cannot tell."
)

Ask = Callable[[str, str], Awaitable[str]]


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


async def ask_claude(question: str, answer: str) -> str:
    """One-word verdict from a small model, with no tools, MCP servers or settings."""
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    options = ClaudeAgentOptions(
        model=JUDGE_MODEL,
        system_prompt=JUDGE_PROMPT,
        tools=[],
        mcp_servers={},
        strict_mcp_config=True,
        setting_sources=[],  # no user hooks: these sessions never reach the wiki
        max_turns=1,
    )
    text = ""
    async for msg in query(prompt=f'Question: "{question}"\nAnswer: "{answer}"', options=options):
        if isinstance(msg, AssistantMessage):
            text += "".join(b.text for b in msg.content if isinstance(b, TextBlock))
    words = re.findall(r"[A-Za-z]+", text)
    return words[0].upper() if words else ""


async def judge_approval(question: str, answer: str | None, ask: Ask = ask_claude,
                         timeout_s: float = JUDGE_TIMEOUT_S) -> bool:
    if not answer or not answer.strip():
        return False
    norm = _normalize(answer)
    if norm in FAST_NO:
        return False
    if is_affirmative(norm):
        return True
    try:
        verdict = await asyncio.wait_for(ask(question, answer), timeout_s)
    except Exception:
        return False
    return verdict == "APPROVE"
