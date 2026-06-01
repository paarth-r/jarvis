import asyncio
import json
import sys
import pytest


async def _run_stub(module: str, timeout: float = 2.0) -> int:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", module,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    msg = json.dumps({"cmd": "shutdown", "params": {}}) + "\n"
    proc.stdin.write(msg.encode())
    await proc.stdin.drain()
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise
    return proc.returncode


@pytest.mark.asyncio
async def test_arm_controller_shuts_down_cleanly():
    code = await _run_stub("jarvis.processes.arm_controller")
    assert code == 0


@pytest.mark.asyncio
async def test_llm_agent_shuts_down_cleanly():
    code = await _run_stub("jarvis.processes.llm_agent")
    assert code == 0
