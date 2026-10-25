"""Tests for the warm Agent SDK brain: streaming, timeouts, resets, permissions."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolPermissionContext,
)

from jarvis.voice.brain import VOICE_PROMPT, Brain, BrainTimeout
from jarvis.voice.config import VoiceConfig


def make_cfg(vault: Path, **kw) -> VoiceConfig:
    return VoiceConfig(
        vault=vault,
        kokoro_model=Path("k.onnx"),
        kokoro_voices=Path("v.bin"),
        smart_turn_model=Path("s.onnx"),
        **kw,
    )


def delta(text: str, parent: str | None = None) -> StreamEvent:
    return StreamEvent(
        uuid="u",
        session_id="abc",
        event={
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        parent_tool_use_id=parent,
    )


def message_start() -> StreamEvent:
    return StreamEvent(uuid="u", session_id="abc", event={"type": "message_start"})


def assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextBlock(text=text)], model="m")


def result(session_id: str = "abc") -> ResultMessage:
    return ResultMessage(
        subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
        num_turns=1, session_id=session_id,
    )


class FakeClient:
    """Scripted stand-in for ClaudeSDKClient."""

    instances: list[FakeClient] = []
    script: list = []
    hang: bool = False
    raise_exc: Exception | None = None

    def __init__(self, options):
        self.options = options
        self.queries: list[str] = []
        self.connected = False
        self.disconnected = False
        FakeClient.instances.append(self)

    async def connect(self, prompt=None):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def query(self, text):
        self.queries.append(text)

    async def receive_response(self):
        if FakeClient.raise_exc is not None:
            raise FakeClient.raise_exc
        if FakeClient.hang:
            await asyncio.sleep(3600)
        for msg in FakeClient.script:
            yield msg


@pytest.fixture(autouse=True)
def reset_fake():
    FakeClient.instances = []
    FakeClient.script = [message_start(), delta("Hello "), delta("there."),
                         assistant("Hello there."), result()]
    FakeClient.hang = False
    FakeClient.raise_exc = None
    yield


@pytest.fixture
def flag_dir(tmp_path):
    return tmp_path / "claude-skip-synthesis"


def make_brain(tmp_path, flag_dir, confirm=None, clock=None, **cfg_kw) -> Brain:
    kwargs = {}
    if clock is not None:
        kwargs["clock"] = clock
    return Brain(
        make_cfg(tmp_path, **cfg_kw), confirm=confirm, client_factory=FakeClient,
        flag_dir=flag_dir, **kwargs,
    )


async def collect(brain: Brain, text: str) -> list[str]:
    return [chunk async for chunk in brain.ask(text)]


async def test_streams_text_deltas(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    assert await collect(brain, "hi") == ["Hello ", "there."]
    assert FakeClient.instances[0].queries == ["hi"]


async def test_falls_back_to_assistant_text_without_partials(tmp_path, flag_dir):
    FakeClient.script = [assistant("Plain reply."), result()]
    brain = make_brain(tmp_path, flag_dir)
    assert await collect(brain, "hi") == ["Plain reply."]


async def test_skips_subagent_text(tmp_path, flag_dir):
    FakeClient.script = [message_start(), delta("inner", parent="t1"),
                         delta("outer"), assistant("outer"), result()]
    brain = make_brain(tmp_path, flag_dir)
    assert await collect(brain, "hi") == ["outer"]


async def test_options(tmp_path, flag_dir):
    room = {"room": object()}
    brain = Brain(make_cfg(tmp_path, model="claude-x"), extra_mcp=room,
                  client_factory=FakeClient, flag_dir=flag_dir)
    await brain.start()
    opts = FakeClient.instances[0].options
    assert opts.cwd == tmp_path
    assert opts.setting_sources == ["user", "project", "local"]
    assert opts.model == "claude-x"
    assert opts.include_partial_messages is True
    assert opts.system_prompt == {
        "type": "preset", "preset": "claude_code", "append": VOICE_PROMPT,
    }
    assert opts.mcp_servers == room


async def test_timeout_raises_and_recycles(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir, turn_timeout_s=0.1)
    await brain.start()
    FakeClient.hang = True
    with pytest.raises(BrainTimeout):
        await collect(brain, "hi")
    first = FakeClient.instances[0]
    FakeClient.hang = False
    assert await collect(brain, "again") == ["Hello ", "there."]
    assert first.queries == ["hi"]
    assert first.disconnected
    assert FakeClient.instances[-1] is not first
    assert FakeClient.instances[-1].queries == ["again"]


async def test_crash_reraises_and_recycles(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    FakeClient.raise_exc = RuntimeError("cli died")
    with pytest.raises(RuntimeError):
        await collect(brain, "hi")
    FakeClient.raise_exc = None
    assert await collect(brain, "again") == ["Hello ", "there."]
    assert len(FakeClient.instances) == 2


async def test_abandoned_turn_recycles(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    gen = brain.ask("hi")
    assert await gen.__anext__() == "Hello "
    await gen.aclose()
    assert await collect(brain, "again") == ["Hello ", "there."]
    assert len(FakeClient.instances) == 2


async def test_idle_reset_swaps_client(tmp_path, flag_dir):
    now = [0.0]
    brain = make_brain(tmp_path, flag_dir, clock=lambda: now[0], idle_reset_s=600.0)
    await brain.start()
    await collect(brain, "hi")
    now[0] = 599.0
    await brain.maybe_reset()
    await collect(brain, "hi")
    assert len(FakeClient.instances) == 1
    now[0] = 599.0 + 600.0
    await brain.maybe_reset()
    await collect(brain, "after idle")
    assert len(FakeClient.instances) == 2
    assert FakeClient.instances[0].disconnected
    assert FakeClient.instances[1].queries == ["after idle"]


async def test_forced_reset_swaps_client(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    await brain.reset()
    await collect(brain, "new conversation")
    assert len(FakeClient.instances) == 2
    assert FakeClient.instances[1].queries == ["new conversation"]


async def test_permission_confirm_path(tmp_path, flag_dir):
    asked: list[str] = []

    async def deny(prompt: str) -> bool:
        asked.append(prompt)
        return False

    brain = make_brain(tmp_path, flag_dir, confirm=deny)
    await brain.start()
    can_use_tool = FakeClient.instances[0].options.can_use_tool
    ctx = ToolPermissionContext()

    res = await can_use_tool("Bash", {"command": "ls"}, ctx)
    assert isinstance(res, PermissionResultDeny)
    assert res.message == "Declined by voice."
    assert len(asked) == 1

    res = await can_use_tool("mcp__room__set_state", {"device": "lamp"}, ctx)
    assert isinstance(res, PermissionResultAllow)
    assert len(asked) == 1

    async def approve(prompt: str) -> bool:
        return True

    brain.confirm = approve
    res = await can_use_tool("Bash", {"command": "ls"}, ctx)
    assert isinstance(res, PermissionResultAllow)


async def test_permission_too_long_denied_without_asking(tmp_path, flag_dir):
    asked: list[str] = []

    async def approve(prompt: str) -> bool:
        asked.append(prompt)
        return True

    brain = make_brain(tmp_path, flag_dir, confirm=approve)
    await brain.start()
    can_use_tool = FakeClient.instances[0].options.can_use_tool
    res = await can_use_tool("Bash", {"command": "echo " + "x" * 400}, ToolPermissionContext())
    assert isinstance(res, PermissionResultDeny)
    assert res.message.startswith("That shell command is too long to confirm by voice.")
    assert asked == []


async def test_permission_denied_without_confirm(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir, confirm=None)
    await brain.start()
    can_use_tool = FakeClient.instances[0].options.can_use_tool
    res = await can_use_tool("Bash", {"command": "ls"}, ToolPermissionContext())
    assert isinstance(res, PermissionResultDeny)


async def test_skip_flag_written(tmp_path, flag_dir):
    FakeClient.script = [SystemMessage(subtype="init", data={"session_id": "abc"}),
                         assistant("ok"), result("abc")]
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    preassigned = FakeClient.instances[0].options.session_id
    assert preassigned and (flag_dir / preassigned).exists()
    await collect(brain, "hi")
    assert (flag_dir / "abc").exists()
    assert brain.session_id == "abc"


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("JARVIS_LIVE") != "1", reason="set JARVIS_LIVE=1")
async def test_live_sdk_smoke(tmp_path, flag_dir):
    from dataclasses import replace

    from claude_agent_sdk import ClaudeSDKClient

    def factory(options):
        return ClaudeSDKClient(replace(options, max_turns=1))

    cfg = make_cfg(tmp_path, turn_timeout_s=120.0)
    brain = Brain(cfg, client_factory=factory, flag_dir=flag_dir)
    await brain.start()
    try:
        text = "".join(await collect(brain, "Reply with just: ok"))
    finally:
        await brain.close()
    assert "ok" in text.lower()
    assert brain.session_id and (flag_dir / brain.session_id).exists()
