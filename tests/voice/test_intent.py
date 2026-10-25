"""Tests for reading approval from a spoken answer."""
from __future__ import annotations

import asyncio
import os

import pytest

from jarvis.voice.intent import ask_claude, judge_approval


def fixed(verdict):
    calls = []

    async def ask(question, answer):
        calls.append(answer)
        return verdict

    ask.calls = calls
    return ask


@pytest.mark.parametrize("answer", ["Yes.", "approved", "Okay!", "sure"])
async def test_obvious_yes_skips_claude(answer):
    ask = fixed("DENY")
    assert await judge_approval("Send it?", answer, ask=ask)
    assert ask.calls == []


@pytest.mark.parametrize("answer", ["No.", "nope", "nevermind", "Wait!", "", None])
async def test_obvious_no_skips_claude(answer):
    ask = fixed("APPROVE")
    assert not await judge_approval("Send it?", answer, ask=ask)
    assert ask.calls == []


async def test_loose_answer_goes_to_claude():
    assert await judge_approval("Send it?", "yea approoved", ask=fixed("APPROVE"))
    assert not await judge_approval("Send it?", "hmm maybe later", ask=fixed("DENY"))
    assert not await judge_approval("Send it?", "uh", ask=fixed("UNCLEAR"))


async def test_errors_and_timeouts_count_as_no():
    async def boom(q, a):
        raise RuntimeError("cli died")

    async def slow(q, a):
        await asyncio.sleep(1)
        return "APPROVE"

    assert not await judge_approval("Send it?", "yea sure go ahead then", ask=boom)
    assert not await judge_approval("Send it?", "yea sure go ahead then", ask=slow, timeout_s=0.05)


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("JARVIS_LIVE") != "1", reason="set JARVIS_LIVE=1")
async def test_live_judge():
    assert await ask_claude("I want to email Sam. Approve?", "yea approoved, send it") == "APPROVE"
    assert await ask_claude("I want to email Sam. Approve?", "actually hold off on that") == "DENY"
