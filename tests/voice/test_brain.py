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

from jarvis.voice.brain import (
    DECLINED,
    NO_CONFIRM,
    TOO_LONG,
    VOICE_PROMPT,
    voice_prompt,
    Brain,
    BrainTimeout,
)
from jarvis.voice.config import VoiceConfig


def make_cfg(vault: Path, **kw) -> VoiceConfig:
    kw.setdefault("permission_mode", "default")  # strict gate unless a test opts in
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
    connect_hang: bool = False
    raise_exc: Exception | None = None
    interrupt_exc: Exception | None = None
    drain_hang: bool = False

    def __init__(self, options):
        self.options = options
        self.queries: list[str] = []
        self.connected = False
        self.disconnected = False
        self.interrupts = 0
        FakeClient.instances.append(self)

    async def connect(self, prompt=None):
        if FakeClient.connect_hang:
            await asyncio.sleep(3600)
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def query(self, text):
        self.queries.append(text)

    async def interrupt(self):
        self.interrupts += 1
        if FakeClient.interrupt_exc is not None:
            raise FakeClient.interrupt_exc

    async def receive_response(self):
        if FakeClient.raise_exc is not None:
            raise FakeClient.raise_exc
        if FakeClient.hang:
            await asyncio.sleep(3600)
        for msg in FakeClient.script:
            if FakeClient.drain_hang and isinstance(msg, ResultMessage):
                await asyncio.sleep(3600)
            yield msg


@pytest.fixture(autouse=True)
def reset_fake():
    FakeClient.instances = []
    FakeClient.script = [message_start(), delta("Hello "), delta("there."),
                         assistant("Hello there."), result()]
    FakeClient.hang = False
    FakeClient.connect_hang = False
    FakeClient.raise_exc = None
    FakeClient.interrupt_exc = None
    FakeClient.drain_hang = False
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
        "type": "preset", "preset": "claude_code", "append": voice_prompt("sir"),
    }
    assert 'Address the user as "sir"' in opts.system_prompt["append"]
    assert opts.mcp_servers == room


def test_voice_prompt_without_address():
    assert voice_prompt("") == VOICE_PROMPT


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


async def test_interrupted_abandon_keeps_session(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    first = FakeClient.instances[0]
    gen = brain.ask("hi")
    assert await gen.__anext__() == "Hello "
    sid = brain.session_id
    await brain.interrupt()
    await gen.aclose()
    assert first.interrupts == 1
    assert not first.disconnected
    assert len(FakeClient.instances) == 1
    assert brain.session_id == sid
    assert await collect(brain, "again") == ["Hello ", "there."]
    assert first.queries == ["hi", "again"]
    assert len(FakeClient.instances) == 1


async def test_interrupt_without_ask_is_noop(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    await brain.interrupt()
    gen = brain.ask("hi")
    assert await gen.__anext__() == "Hello "
    await gen.aclose()
    await asyncio.sleep(0.01)
    assert FakeClient.instances[0].interrupts == 0
    assert len(FakeClient.instances) == 2


async def test_interrupt_mark_does_not_leak_to_next_ask(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    gen = brain.ask("hi")
    await gen.__anext__()
    await brain.interrupt()
    await gen.aclose()
    gen = brain.ask("again")
    await gen.__anext__()
    await gen.aclose()
    await asyncio.sleep(0.01)
    assert FakeClient.instances[0].disconnected
    assert len(FakeClient.instances) == 2


async def test_interrupt_drain_timeout_recycles(tmp_path, flag_dir, monkeypatch):
    monkeypatch.setattr("jarvis.voice.brain.INTERRUPT_DRAIN_TIMEOUT_S", 0.05)
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    first = FakeClient.instances[0]
    FakeClient.drain_hang = True
    gen = brain.ask("hi")
    await gen.__anext__()
    await brain.interrupt()
    await gen.aclose()
    FakeClient.drain_hang = False
    assert first.interrupts == 1
    assert await collect(brain, "again") == ["Hello ", "there."]
    assert first.disconnected
    assert len(FakeClient.instances) == 2


async def test_interrupt_failure_recycles(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    FakeClient.interrupt_exc = RuntimeError("cli gone")
    gen = brain.ask("hi")
    await gen.__anext__()
    await brain.interrupt()
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
    assert res.message == TOO_LONG
    assert res.message.startswith("That action is too long to confirm by voice.")
    assert "shell command" in res.message
    assert asked == []


async def test_permission_denied_without_confirm(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir, confirm=None)
    await brain.start()
    can_use_tool = FakeClient.instances[0].options.can_use_tool
    res = await can_use_tool("Bash", {"command": "ls"}, ToolPermissionContext())
    assert isinstance(res, PermissionResultDeny)
    assert res.message == NO_CONFIRM


async def test_permission_not_asked_is_not_a_decline(tmp_path, flag_dir):
    async def not_asked(prompt: str) -> None:
        return None  # the loop had no active turn, so nobody was asked

    brain = make_brain(tmp_path, flag_dir, confirm=not_asked)
    await brain.start()
    can_use_tool = FakeClient.instances[0].options.can_use_tool
    res = await can_use_tool("Bash", {"command": "ls"}, ToolPermissionContext())
    assert isinstance(res, PermissionResultDeny)
    assert res.message == NO_CONFIRM
    hook = pre_tool_use_hook(FakeClient.instances[0].options)
    out = await hook(hook_input("Bash", {"command": "ls"}), "t1", {"signal": None})
    assert decision(out) == ("deny", NO_CONFIRM)


# -- PreToolUse hook: the policy must hold even when settings allow a tool ----

IMESSAGE_REPLY = "mcp__plugin_imessage_imessage__reply"


def pre_tool_use_hook(options):
    matchers = options.hooks["PreToolUse"]
    assert len(matchers) == 1 and len(matchers[0].hooks) == 1
    return matchers[0].hooks[0]


def hook_input(name: str, tool_input: dict, tool_use_id: str = "t1") -> dict:
    return {
        "hook_event_name": "PreToolUse", "session_id": "s", "transcript_path": "",
        "cwd": "", "tool_name": name, "tool_input": tool_input,
        "tool_use_id": tool_use_id,
    }


def decision(out: dict) -> tuple[str, str | None] | None:
    specific = out.get("hookSpecificOutput")
    if not specific or "permissionDecision" not in specific:
        return None
    assert specific["hookEventName"] == "PreToolUse"
    return specific["permissionDecision"], specific.get("permissionDecisionReason")


async def test_pre_tool_use_hook_registered_for_every_tool(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir, turn_timeout_s=180.0)
    await brain.start()
    matcher = FakeClient.instances[0].options.hooks["PreToolUse"][0]
    assert matcher.matcher is None  # every tool, not a name pattern
    # The hook must outlive the turn: a hook timeout would fall through to
    # settings allow rules, so the turn timeout has to fire first.
    assert matcher.timeout is not None and matcher.timeout > 180.0


async def test_hook_asks_for_a_settings_allowlisted_tool(tmp_path, flag_dir):
    asked: list[str] = []

    async def approve(prompt: str) -> bool:
        asked.append(prompt)
        return True

    brain = make_brain(tmp_path, flag_dir, confirm=approve, self_chat_id="self-chat")
    await brain.start()
    options = FakeClient.instances[0].options
    hook = pre_tool_use_hook(options)
    tool_input = {"chat_id": "someone-else", "text": "running late"}

    out = await hook(hook_input(IMESSAGE_REPLY, tool_input, "t1"), "t1", {"signal": None})
    assert decision(out) == ("allow", None)
    assert len(asked) == 1 and "someone-else" in asked[0]

    # The same tool call reaching the backstop is never asked about twice.
    res = await options.can_use_tool(
        IMESSAGE_REPLY, tool_input, ToolPermissionContext(tool_use_id="t1"))
    assert isinstance(res, PermissionResultAllow)
    assert len(asked) == 1


async def test_hook_denies_when_declined(tmp_path, flag_dir):
    async def deny(prompt: str) -> bool:
        return False

    brain = make_brain(tmp_path, flag_dir, confirm=deny, self_chat_id="self-chat")
    await brain.start()
    hook = pre_tool_use_hook(FakeClient.instances[0].options)
    out = await hook(hook_input(IMESSAGE_REPLY, {"chat_id": "x", "text": "hi"}),
                     "t1", {"signal": None})
    assert decision(out) == ("deny", DECLINED)


async def test_hook_denies_too_long_without_asking(tmp_path, flag_dir):
    asked: list[str] = []

    async def approve(prompt: str) -> bool:
        asked.append(prompt)
        return True

    brain = make_brain(tmp_path, flag_dir, confirm=approve)
    await brain.start()
    hook = pre_tool_use_hook(FakeClient.instances[0].options)
    out = await hook(hook_input("Bash", {"command": "echo " + "x" * 400}),
                     "t1", {"signal": None})
    assert decision(out) == ("deny", TOO_LONG)
    assert asked == []


async def test_hook_leaves_policy_allowed_tools_alone(tmp_path, flag_dir):
    asked: list[str] = []

    async def approve(prompt: str) -> bool:
        asked.append(prompt)
        return True

    brain = make_brain(tmp_path, flag_dir, confirm=approve)
    await brain.start()
    hook = pre_tool_use_hook(FakeClient.instances[0].options)
    out = await hook(hook_input("Read", {"file_path": "/etc/hosts"}), "t1", {"signal": None})
    assert decision(out) is None  # no decision: settings deny rules still apply
    assert asked == []


async def test_hook_denies_when_confirm_raises(tmp_path, flag_dir):
    async def broken(prompt: str) -> bool:
        raise RuntimeError("mic gone")

    brain = make_brain(tmp_path, flag_dir, confirm=broken)
    await brain.start()
    hook = pre_tool_use_hook(FakeClient.instances[0].options)
    out = await hook(hook_input("Bash", {"command": "ls"}), "t1", {"signal": None})
    assert decision(out)[0] == "deny"


# -- shutdown -----------------------------------------------------------------

async def test_close_disconnects_a_client_whose_connect_was_cancelled(tmp_path, flag_dir):
    FakeClient.connect_hang = True
    brain = make_brain(tmp_path, flag_dir)
    starting = asyncio.create_task(brain.start())
    await asyncio.sleep(0.01)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await brain.close()
    assert FakeClient.instances[0].disconnected


async def test_close_survives_a_cancelled_warm_up(tmp_path, flag_dir):
    brain = make_brain(tmp_path, flag_dir)
    await brain.start()
    first = FakeClient.instances[0]
    FakeClient.connect_hang = True
    await brain.reset()  # starts a background warm-up that hangs in connect
    await asyncio.sleep(0.01)
    brain._warming.cancel()
    await brain.close()  # must not raise CancelledError, must still clean up
    assert first.disconnected
    assert FakeClient.instances[1].disconnected


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


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("JARVIS_LIVE") != "1", reason="set JARVIS_LIVE=1")
async def test_live_hook_beats_a_settings_allow_rule(tmp_path, flag_dir):
    """A tool allowlisted in settings still needs a spoken yes (via the hook)."""
    import json
    from dataclasses import replace

    from claude_agent_sdk import ClaudeSDKClient

    settings = tmp_path / ".claude" / "settings.local.json"
    settings.parent.mkdir()
    settings.write_text(json.dumps({"permissions": {"allow": ["Bash(echo *)"]}}))
    asked: list[str] = []

    async def decline(prompt: str) -> bool:
        asked.append(prompt)
        return False

    def factory(options):
        return ClaudeSDKClient(replace(options, max_turns=3))

    cfg = make_cfg(tmp_path, turn_timeout_s=120.0)
    brain = Brain(cfg, confirm=decline, client_factory=factory, flag_dir=flag_dir)
    await brain.start()
    try:
        await collect(brain, "Use the Bash tool to run exactly this command: "
                             "echo jarvis-hook-check. Then reply with one word.")
    finally:
        await brain.close()
    # The command is never read aloud; the hook asked with Claude's summary.
    assert any(q.startswith("I want to") and q.endswith("Approve?") for q in asked), asked


async def test_auto_mode_passes_through_and_skips_shell_approval(tmp_path, flag_dir):
    asked: list[str] = []

    async def approve(prompt: str) -> bool:
        asked.append(prompt)
        return True

    brain = make_brain(tmp_path, flag_dir, confirm=approve, permission_mode="auto")
    await brain.start()
    opts = FakeClient.instances[0].options
    assert opts.permission_mode == "auto"
    hook = pre_tool_use_hook(opts)
    out = await hook(hook_input("Bash", {"command": "ls", "description": "List files"}), "t1", {"signal": None})
    assert decision(out) is None or decision(out)[0] != "deny"
    assert asked == []
    out = await hook(hook_input("mcp__claude_ai_Gmail__send_message", {"to": ["a@x.com"]}), "t2", {"signal": None})
    assert decision(out)[0] == "allow" and len(asked) == 1
