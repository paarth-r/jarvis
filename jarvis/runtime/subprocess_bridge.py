from __future__ import annotations
import asyncio
import json
import logging
import sys
from typing import Optional

from .event_bus import EventBus

logger = logging.getLogger(__name__)


class SubprocessBridge:
    """
    Spawns a child process, exchanges JSON-lines over stdin/stdout.
    stderr is inherited so child stack traces appear in the parent terminal.
    Restarts the child on non-zero exit with exponential backoff.
    """

    def __init__(self, bridge_id: str, module_path: str, bus: EventBus) -> None:
        self._bridge_id = bridge_id
        self._module_path = module_path
        self._bus = bus
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._backoff = 1.0

    async def run(self) -> None:
        while True:
            try:
                await self._run_once()
                # clean exit — don't restart
                logger.info("Bridge %s exited cleanly", self._bridge_id)
                return
            except asyncio.CancelledError:
                await self._shutdown()
                raise
            except Exception as exc:
                logger.error("Bridge %s crashed: %s", self._bridge_id, exc)

            logger.info("Bridge %s restarting in %.1fs", self._bridge_id, self._backoff)
            await asyncio.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, 30.0)

    async def _run_once(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", self._module_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=None,   # inherit — child stack traces appear in parent terminal
        )
        self._backoff = 1.0

        async for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                await self._dispatch(data)
            except json.JSONDecodeError:
                logger.warning("Bridge %s bad JSON: %r", self._bridge_id, line)

        await self._proc.wait()
        if self._proc.returncode != 0:
            raise RuntimeError(f"Process exited with code {self._proc.returncode}")

    async def _dispatch(self, data: dict) -> None:
        # Extended in future tasks when processes emit real events
        pass

    async def send(self, cmd: str, params: dict | None = None) -> None:
        if self._proc and self._proc.stdin:
            msg = json.dumps({"cmd": cmd, "params": params or {}}) + "\n"
            self._proc.stdin.write(msg.encode())
            await self._proc.stdin.drain()

    async def _shutdown(self) -> None:
        if self._proc:
            try:
                await self.send("shutdown")
                await asyncio.wait_for(self._proc.wait(), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                self._proc.kill()
            self._proc = None
