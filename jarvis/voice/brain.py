"""The brain: a warm, long-lived Claude Agent SDK session the voice loop talks to.

The client is connected at boot and kept warm because claude.ai connectors
(Gmail, Calendar) only finish connecting several seconds after session start.
After a reset, timeout or crash the old client is dropped and a replacement is
pre-warmed in the background, so the next turn rarely pays the startup cost.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolPermissionContext,
)

from jarvis.voice.config import VoiceConfig
from jarvis.voice.permissions import classify_tool, describe_action, too_long_to_confirm

log = logging.getLogger(__name__)

VOICE_PROMPT = (
    "You are being spoken to through a voice interface in the user's room and your "
    "reply will be read aloud. Answer in short, natural spoken sentences. No "
    "markdown, lists, tables, code or URLs. Say numbers and times the way a person "
    "would. If the answer is long (a plan, a diff, a table, more than about four "
    "sentences), write it to the wiki or send it to the user's phone over iMessage, "
    "then say in one sentence where you put it. Room devices are controlled with "
    "the room tools."
)
DECLINED = "Declined by voice."
NO_CONFIRM = "No voice confirmation is available, so this action was not run."
TOO_LONG = (
    "That shell command is too long to confirm by voice. Split it into shorter "
    "commands or write it to a script in the vault and run that."
)
DISCONNECT_TIMEOUT_S = 10.0


class BrainTimeout(Exception):
    """A turn produced nothing for longer than the configured turn timeout."""


class Brain:
    def __init__(
        self,
        cfg: VoiceConfig,
        confirm: Callable[[str], Awaitable[bool]] | None = None,
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
        self._warming: asyncio.Task | None = None
        self._background: set[asyncio.Task] = set()
        self._last_active = clock()

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
        """Disconnect everything (shutdown)."""
        if self._warming is not None:
            task, self._warming = self._warming, None
            with contextlib.suppress(Exception):
                self._client = await task
        client, self._client = self._client, None
        if client is not None:
            await self._disconnect(client)
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)

    # -- turns -------------------------------------------------------------

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
        try:
            budget = await self._timed(client.query(text), budget)
            stream = client.receive_response().__aiter__()
            streamed = False  # current message already produced text deltas
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
                elif isinstance(msg, AssistantMessage):
                    if msg.parent_tool_use_id is not None or streamed:
                        continue
                    for block in msg.content:
                        if isinstance(block, TextBlock) and block.text:
                            yield block.text
            finished = True
        except TimeoutError:
            raise BrainTimeout(
                f"no reply within {self.cfg.turn_timeout_s:.0f} s"
            ) from None
        finally:
            self._last_active = self._clock()
            if not finished and self._client is client:
                log.warning("brain turn did not complete, recycling the client")
                self._recycle()

    async def _timed(self, awaitable: Awaitable[Any], budget: float) -> float:
        start = time.monotonic()
        await asyncio.wait_for(awaitable, max(budget, 0.0))
        return budget - (time.monotonic() - start)

    async def _timed_next(self, stream: AsyncIterator[Any], budget: float):
        start = time.monotonic()
        msg = await asyncio.wait_for(anext(stream), max(budget, 0.0))
        return msg, budget - (time.monotonic() - start)

    # -- permissions -------------------------------------------------------

    async def _can_use_tool(
        self, name: str, tool_input: dict, ctx: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        verdict = classify_tool(name, tool_input, self.cfg.vault, self.cfg.self_chat_id)
        if verdict == "allow":
            return PermissionResultAllow()
        if too_long_to_confirm(name, tool_input):
            return PermissionResultDeny(message=TOO_LONG)
        if self.confirm is None:
            return PermissionResultDeny(message=NO_CONFIRM)
        try:
            approved = await self.confirm(describe_action(name, tool_input))
        except Exception:
            log.exception("voice confirmation failed; treating as declined")
            approved = False
        return PermissionResultAllow() if approved else PermissionResultDeny(message=DECLINED)

    # -- client management -------------------------------------------------

    def _options(self, session_id: str) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            cwd=self.cfg.vault,
            setting_sources=["user", "project", "local"],
            model=self.cfg.model,
            system_prompt={"type": "preset", "preset": "claude_code", "append": VOICE_PROMPT},
            mcp_servers=self._extra_mcp or {},
            include_partial_messages=True,
            can_use_tool=self._can_use_tool,
            session_id=session_id,
        )

    async def _connect(self) -> Any:
        # Pre-assign the session id so the skip-synthesis flag exists before the
        # CLI starts; SessionEnd fires even for a session that never got a turn.
        session_id = str(uuid.uuid4())
        self._record_session(session_id)
        client = self._client_factory(self._options(session_id))
        await client.connect()
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
