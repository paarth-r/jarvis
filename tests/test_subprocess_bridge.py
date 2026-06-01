import asyncio
import sys
import pytest
from jarvis.runtime.event_bus import EventBus
from jarvis.runtime.subprocess_bridge import SubprocessBridge


# A fake subprocess that emits one JSON event then exits cleanly
FAKE_PROC_SCRIPT = """
import sys, json, time
print(json.dumps({"type": "ping", "payload": {"msg": "hello"}, "timestamp": 1.0}), flush=True)
# Wait for shutdown command
for line in sys.stdin:
    data = json.loads(line)
    if data.get("cmd") == "shutdown":
        break
sys.exit(0)
"""


@pytest.mark.asyncio
async def test_bridge_restarts_on_nonzero_exit():
    bus = EventBus()
    start_count = [0]
    original_run_once = SubprocessBridge._run_once

    async def counting_run_once(self):
        start_count[0] += 1
        if start_count[0] < 3:
            raise RuntimeError("simulated crash")
        # On 3rd attempt, cancel to stop the loop
        raise asyncio.CancelledError()

    SubprocessBridge._run_once = counting_run_once
    bridge = SubprocessBridge("test", "fake.module", bus)
    bridge._backoff = 0.001  # speed up test

    try:
        await asyncio.wait_for(bridge.run(), timeout=1.0)
    except asyncio.CancelledError:
        pass
    finally:
        SubprocessBridge._run_once = original_run_once

    assert start_count[0] == 3


@pytest.mark.asyncio
async def test_bridge_send_writes_json():
    bus = EventBus()
    bridge = SubprocessBridge("test", "fake.module", bus)

    written = []

    class _FakeStdin:
        async def drain(self): pass
        def write(self, data): written.append(data)

    class _FakeProc:
        stdin = _FakeStdin()

    bridge._proc = _FakeProc()
    await bridge.send("shutdown", {"reason": "test"})

    import json
    assert len(written) == 1
    msg = json.loads(written[0].decode())
    assert msg["cmd"] == "shutdown"
    assert msg["params"]["reason"] == "test"
