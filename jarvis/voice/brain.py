"""The brain: a warm, long-lived Claude Agent SDK session the voice loop talks to.

The client is connected at boot and kept warm because claude.ai connectors
(Gmail, Calendar) only finish connecting several seconds after session start.
After a reset, timeout or crash the old client is dropped and a replacement is
pre-warmed in the background, so the next turn rarely pays the startup cost.

The voice permission policy is enforced in a PreToolUse hook, not only in
``can_use_tool``: the CLI applies allow rules from settings files before it
consults ``can_use_tool``, so a tool allowlisted in the user's settings would
otherwise run without a spoken yes. Hooks run for every tool call regardless of
permission rules. ``can_use_tool`` stays as a backstop with the same policy.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookContext,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from jarvis.voice.config import VoiceConfig
from jarvis.voice.permissions import classify_tool, describe_action, too_long_to_confirm
from jarvis.voice.trace import tool_summary, trace

log = logging.getLogger(__name__)

VOICE_PROMPT = (
    "You are being spoken to through a voice interface in the user's room and your "
    "reply will be read aloud. Keep it short and conversational, like talking to "
    "someone in the room: usually one or two sentences, answer first, no preamble, "
    "no recap of the question, no closing offers of more help. Go longer only when "
    "they ask for detail. Don't narrate your work; at most one brief line such as "
    "'One moment, sir' before a slow lookup, then just the answer. If they are "
    "backing out or brushing you off (nevermind, forget it, actually no, it's "
    "fine), reply with only 'Okay.' and do nothing else. If they are just wrapping up "
    "(thanks, sounds good, cheers, that's all), reply with only 'You're welcome.' "
    "or 'Okay.'; both end the conversation. No "
    "markdown, lists, tables, code or URLs. Say numbers and times the way a person "
    "would. If the answer is long (a plan, a diff, a table, more than about four "
    "sentences), write it to the wiki or send it to the user's phone over iMessage, "
    "then say in one sentence where you put it. Room devices are controlled with "
    "the room tools. Never read out commands, code, file paths or tool names. Some "
    "actions need the user's spoken approval, and they hear a request, not a command: "
    "whenever you run a shell command, give the Bash tool a brief plain-English "
    "summary of what you are trying to do, about ten words, five seconds spoken "
    "(for example 'delete the old logs folder'). They hear 'I want to <summary>. "
    "Approve?' and the command runs only if they approve."
)


def voice_prompt(address_as: str) -> str:
    """VOICE_PROMPT plus how to address the user (e.g. "sir"), if configured."""
    if not address_as:
        return VOICE_PROMPT
    return (
        f'{VOICE_PROMPT} Address the user as "{address_as}", the way a calm, dry, '
        f'British butler would, for example "Good evening, {address_as}."'
    )


DECLINED = "Declined by voice."
NO_CONFIRM = "No voice confirmation is available, so this action was not run."
TOO_LONG = (
    "That action is too long to confirm by voice. If it is a shell command, give "
    "the Bash tool a short plain-English description of the change, or split it "
    "into shorter commands; "
    "otherwise make it smaller, for example fewer recipients."
)
POLICY_ERROR = "The voice permission check failed, so this action was not run."
DISCONNECT_TIMEOUT_S = 10.0
INTERRUPT_DRAIN_TIMEOUT_S = 5.0
# The hook must outlive the turn: if it timed out first, the CLI would fall back
# to settings allow rules. The turn timeout fires first and recycles the client.
HOOK_TIMEOUT_MARGIN_S = 60.0
MAX_APPROVED_IDS = 256


class BrainTimeout(Exception):
    """A turn produced nothing for longer than the configured turn timeout."""


class Brain:
    def __init__(
        self,
        cfg: VoiceConfig,
        confirm: Callable[[str], Awaitable[bool | None]] | None = None,
        extra_mcp: dict | None = None,
        client_factory: Callable[[ClaudeAgentOptions], Any] = ClaudeSDKClient,
        flag_dir: Path = Path("/tmp/claude-skip-synthesis"),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.confirm = confirm
        self.session_id: str | None = None
        self._extra_mcp = extra_mcp
        self._client_factory = client_factory
        self._flag_dir = Path(flag_dir)
        self._clock = clock
        self._client: Any | None = None
        self._connecting: set[Any] = set()  # clients whose connect() never finished
        self._approved: OrderedDict[str, None] = OrderedDict()  # tool_use_ids said yes to
        self._warming: asyncio.Task | None = None
        self._background: set[asyncio.Task] = set()
        self._last_active = clock()
        self._in_flight = False
        self._interrupted = False

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Connect a client now so it is warm before the first turn."""
        await self._ready()
        self._last_active = self._clock()

    async def maybe_reset(self) -> None:
        """Start a fresh conversation if the current one has been idle too long."""
        if self._client is None:
            return
        if self._clock() - self._last_active >= self.cfg.idle_reset_s:
            log.info("brain idle for %.0f s, starting a fresh session",
                     self._clock() - self._last_active)
            await self.reset()

    async def reset(self) -> None:
        """Drop the current conversation and pre-warm a fresh client."""
        self._recycle()

    async def close(self) -> None:
        """Disconnect everything (shutdown).

        Safe to call while shutting down on Ctrl-C: a cancelled warm-up or an
        interrupted connect never skips the disconnects. If this task itself is
        being cancelled, the cancellation is re-raised after cleanup.
        """
        clients: list[Any] = []
        if self._warming is not None:
            task, self._warming = self._warming, None
            try:
                clients.append(await task)
            except (Exception, asyncio.CancelledError) as exc:
                # A cancelled warm-up raises CancelledError here even when this
                # task is not being cancelled; either way, keep cleaning up.
                log.debug("brain warm-up ended without a client: %r", exc)
        client, self._client = self._client, None
        if client is not None:
            clients.append(client)
        clients += [c for c in self._connecting if c not in clients]
        self._connecting.clear()
        for c in clients:
            await self._disconnect(c)
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError

    # -- turns -------------------------------------------------------------

    async def interrupt(self) -> None:
        """Mark the in-flight ask interrupted; closing it then keeps the session."""
        if self._in_flight:
            self._interrupted = True

    async def ask(self, text: str) -> AsyncIterator[str]:
        """Send one user turn and yield the reply's text as it streams.

        The turn timeout budgets time spent waiting on Claude, not time the
        caller spends consuming chunks. On timeout (``BrainTimeout``), crash
        (re-raised) or an abandoned iteration, the client is recycled.
        """
        client = await self._ready()
        self._last_active = self._clock()
        budget = self.cfg.turn_timeout_s
        finished = False
        stream = None
        self._in_flight, self._interrupted = True, False
        try:
            budget = await self._timed(client.query(text), budget)
            stream = client.receive_response().__aiter__()
            streamed = False  # current message already produced text deltas
            thought = ""  # thinking deltas of the current block, traced when it ends
            while True:
                try:
                    msg, budget = await self._timed_next(stream, budget)
                except StopAsyncIteration:
                    break
                if isinstance(msg, SystemMessage):
                    if msg.subtype == "init":
                        self._record_session(msg.data.get("session_id"))
                elif isinstance(msg, ResultMessage):
                    self._record_session(msg.session_id)
                elif isinstance(msg, StreamEvent):
                    if msg.parent_tool_use_id is not None:
                        continue
                    event = msg.event
                    if event.get("type") == "message_start":
                        streamed = False
                    elif event.get("type") == "content_block_delta":
                        d = event.get("delta") or {}
                        if d.get("type") == "text_delta" and d.get("text"):
                            streamed = True
                            yield d["text"]
                        elif d.get("type") == "thinking_delta":
                            thought += d.get("thinking", "")
                    elif event.get("type") == "content_block_stop" and thought:
                        trace("thinking", thought, limit=1200)
                        thought = ""
                elif isinstance(msg, AssistantMessage):
                    sub = msg.parent_tool_use_id is not None
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            trace("subagent tool" if sub else "tool",
                                  tool_summary(block.name, block.input or {}))
                    if sub or streamed:
                        continue
                    for block in msg.content:
                        if isinstance(block, ThinkingBlock) and block.thinking:
                            trace("thinking", block.thinking, limit=1200)
                        elif isinstance(block, TextBlock) and block.text:
                            yield block.text
                elif isinstance(msg, UserMessage) and msg.parent_tool_use_id is None:
                    for block in msg.content if isinstance(msg.content, list) else []:
                        if isinstance(block, ToolResultBlock):
                            body = block.content
                            if isinstance(body, list):
                                body = " ".join(str(c.get("text", "")) for c in body if isinstance(c, dict))
                            trace("result", ("error: " if block.is_error else "") + str(body or "ok"),
                                  limit=200)
            finished = True
        except TimeoutError:
            raise BrainTimeout(
                f"no reply within {self.cfg.turn_timeout_s:.0f} s"
            ) from None
        finally:
            interrupted, self._in_flight, self._interrupted = self._interrupted, False, False
            if (not finished and interrupted and stream is not None
                    and self._client is client):
                finished = await self._interrupt_and_drain(client, stream)
            self._last_active = self._clock()
            if not finished and self._client is client:
                log.warning("brain turn did not complete, recycling the client")
                self._recycle()

    async def _interrupt_and_drain(self, client: Any, stream: AsyncIterator[Any]) -> bool:
        """Stop the turn on the CLI and read to its ResultMessage. True if clean."""
        async def drain() -> None:
            await client.interrupt()
            async for msg in stream:
                if isinstance(msg, ResultMessage):
                    break

        try:
            await asyncio.wait_for(drain(), INTERRUPT_DRAIN_TIMEOUT_S)
            return True
        except Exception:
            log.warning("could not interrupt the turn cleanly", exc_info=True)
            return False

    async def _timed(self, awaitable: Awaitable[Any], budget: float) -> float:
        start = time.monotonic()
        await asyncio.wait_for(awaitable, max(budget, 0.0))
        return budget - (time.monotonic() - start)

    async def _timed_next(self, stream: AsyncIterator[Any], budget: float):
        start = time.monotonic()
        msg = await asyncio.wait_for(anext(stream), max(budget, 0.0))
        return msg, budget - (time.monotonic() - start)

    # -- permissions -------------------------------------------------------

    def _needs_confirm(self, name: str, tool_input: dict) -> bool:
        verdict = classify_tool(name, tool_input, self.cfg.vault, self.cfg.self_chat_id,
                                auto=self.cfg.permission_mode == "auto")
        return verdict != "allow"

    async def _confirm_action(self, name: str, tool_input: dict) -> str | None:
        """Ask by voice. None if approved, else the reason the call is denied."""
        if too_long_to_confirm(name, tool_input):
            return TOO_LONG
        if self.confirm is None:
            return NO_CONFIRM
        try:
            approved = await self.confirm(describe_action(name, tool_input))
        except Exception:
            log.exception("voice confirmation failed; treating as declined")
            return DECLINED
        if approved is None:  # nobody was asked (no active turn)
            return NO_CONFIRM
        return None if approved else DECLINED

    async def _pre_tool_use(
        self, input_data: Any, tool_use_id: str | None, context: HookContext
    ) -> dict[str, Any]:
        """PreToolUse hook: the policy, evaluated before settings allow rules.

        Policy-allowed tools get no decision, so the CLI's own rules (including
        the user's deny rules) still apply. Anything else is decided here.
        Never raises: a failing hook would fall through to settings rules.
        """
        try:
            name = str(input_data.get("tool_name", ""))
            tool_input = input_data.get("tool_input") or {}
            if not self._needs_confirm(name, tool_input):
                return {}
            denial = await self._confirm_action(name, tool_input)
        except Exception:
            log.exception("voice permission hook failed; denying")
            denial = POLICY_ERROR
        if denial is not None:
            return _hook_decision("deny", denial)
        self._remember_approved(tool_use_id or input_data.get("tool_use_id"))
        return _hook_decision("allow")

    async def _can_use_tool(
        self, name: str, tool_input: dict, ctx: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Backstop with the same policy; never asks twice for one tool call."""
        if ctx.tool_use_id is not None and ctx.tool_use_id in self._approved:
            del self._approved[ctx.tool_use_id]  # the hook already got a spoken yes
            return PermissionResultAllow()
        if not self._needs_confirm(name, tool_input):
            return PermissionResultAllow()
        denial = await self._confirm_action(name, tool_input)
        if denial is not None:
            return PermissionResultDeny(message=denial)
        return PermissionResultAllow()

    def _remember_approved(self, tool_use_id: str | None) -> None:
        if not tool_use_id:
            return
        self._approved[tool_use_id] = None
        while len(self._approved) > MAX_APPROVED_IDS:
            self._approved.popitem(last=False)

    # -- client management -------------------------------------------------

    def _options(self, session_id: str) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            cwd=self.cfg.vault,
            setting_sources=["user", "project", "local"],
            model=self.cfg.model,
            system_prompt={"type": "preset", "preset": "claude_code", "append": voice_prompt(self.cfg.address_as)},
            mcp_servers=self._extra_mcp or {},
            include_partial_messages=True,
            permission_mode=self.cfg.permission_mode,  # type: ignore[arg-type]
            can_use_tool=self._can_use_tool,
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=[self._pre_tool_use],
                        timeout=self.cfg.turn_timeout_s + HOOK_TIMEOUT_MARGIN_S,
                    )
                ]
            },
            session_id=session_id,
        )

    async def _connect(self) -> Any:
        # Pre-assign the session id so the skip-synthesis flag exists before the
        # CLI starts; SessionEnd fires even for a session that never got a turn.
        session_id = str(uuid.uuid4())
        self._record_session(session_id)
        client = self._client_factory(self._options(session_id))
        self._connecting.add(client)
        try:
            await client.connect()
        except Exception:
            self._connecting.discard(client)
            self._spawn(self._disconnect(client))
            raise
        # On cancellation the client stays tracked so close() disconnects it.
        self._connecting.discard(client)
        return client

    async def _ready(self) -> Any:
        if self._warming is not None:
            task, self._warming = self._warming, None
            try:
                self._client = await task
            except Exception:
                log.exception("background brain warm-up failed; retrying now")
                self._client = None
        if self._client is None:
            self._client = await self._connect()
        return self._client

    def _recycle(self) -> None:
        old, self._client = self._client, None
        if old is not None:
            self._spawn(self._disconnect(old))
        if self._warming is None:
            self._warming = asyncio.create_task(self._connect())
        self._last_active = self._clock()

    def _spawn(self, coro: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _disconnect(self, client: Any) -> None:
        try:
            await asyncio.wait_for(client.disconnect(), DISCONNECT_TIMEOUT_S)
        except Exception:
            log.warning("brain client did not disconnect cleanly", exc_info=True)

    def _record_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        self.session_id = session_id
        try:
            self._flag_dir.mkdir(parents=True, exist_ok=True)
            (self._flag_dir / session_id).touch()
        except OSError:
            log.warning("could not write skip-synthesis flag for %s", session_id)


def _hook_decision(decision: str, reason: str | None = None) -> dict[str, Any]:
    specific: dict[str, Any] = {"hookEventName": "PreToolUse", "permissionDecision": decision}
    if reason is not None:
        specific["permissionDecisionReason"] = reason
    return {"hookSpecificOutput": specific}
