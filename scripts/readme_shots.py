"""Render the README screenshots of the gesture studio (docs/images/*.png).

No camera needed: the real studio server and the real gesture engine run on a
bus fed with synthetic hands (tests/hand_synth.py). Custom gestures are made
through the real HTTP API into a temporary library, so ~/.jarvis is never
touched, and the cursor and key backends are stubbed so nothing is moved or
pressed. Headless Chrome is driven over the DevTools protocol to capture:

  studio.png      the live view with a gesture held, plus the gesture list
  studio-add.png  the add-gesture flow with its quality check

Run from the repo root:  python scripts/readme_shots.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = ROOT / "docs" / "images"
W, H = 1440, 900

# Never move the real mouse: swap the Quartz controller for a stub before import.
_stub = ModuleType("mouse_control.control")
_stub.MouseController = lambda *a, **k: MagicMock()
sys.modules["mouse_control.control"] = _stub

from jarvis.events import WorldPoseEvent  # noqa: E402
from jarvis.modules.gesture_hci import GestureHCIModule  # noqa: E402
from jarvis.runtime.event_bus import EventBus  # noqa: E402
from jarvis.studio.module import StudioModule  # noqa: E402
from tests.hand_synth import OPEN, PEACE, ROCK, THREE, hand, rotation  # noqa: E402
from tests.test_gesture_actions import FakeBackend  # noqa: E402

FOUR = {**THREE, "pinky": (0.0, 0.0, 0.0)}
GESTURES = [  # name, pose, binding
    ("peace", PEACE, {"type": "media", "key": "play_pause"}),
    ("rock", ROCK, {"type": "keys", "combo": "cmd+shift+4"}),
    ("three", THREE, {"type": "macro", "steps": [{"type": "text", "text": "on my way"},
                                                 {"type": "wait", "seconds": 0.3},
                                                 {"type": "keys", "combo": "return"}]}),
]


class Feeder:
    """Publishes a synthetic hand at 30 fps with slight natural motion."""

    def __init__(self, bus):
        self.bus, self.pose, self.seed, self.t0 = bus, OPEN, 0, time.perf_counter()

    async def run(self):
        rng = np.random.default_rng(1)
        while True:
            t = time.perf_counter() - self.t0
            R = rotation(0.18 * np.sin(0.9 * t) - 0.1, 0.25 * np.sin(0.6 * t + self.seed) + 0.1,
                         0.08 * np.sin(1.3 * t))
            await self.bus.publish(WorldPoseEvent(
                landmarks_3d=np.zeros((21, 3)), timestamp=time.perf_counter(), confidence=1.0,
                hand_side="right", hand_shape=hand(self.pose, R=R, offset=(0.0, 0.1, 0.0), noise=0.0015, rng=rng)))
            await asyncio.sleep(1 / 30)


def http(port, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                 headers={"X-Jarvis-Studio": "1", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read() or "null")


class CDP:
    """A just-enough DevTools client (stdlib websocket) so no extra deps."""

    def __init__(self, ws_url):
        host, port, path = ws_url[5:].split("/", 1)[0].rsplit(":", 1) + ["/" + ws_url[5:].split("/", 1)[1]]
        self.s = socket.create_connection((host, int(port)))
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                        f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                        "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(1)
        self.n = 0

    def _read(self, n):
        buf = b""
        while len(buf) < n:
            buf += self.s.recv(n - len(buf))
        return buf

    def _recv(self):
        msg = b""
        while True:
            b0, b1 = self._read(2)
            ln = b1 & 0x7F
            if ln == 126:
                ln = struct.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = struct.unpack(">Q", self._read(8))[0]
            msg += self._read(ln)
            if b0 & 0x80:
                return json.loads(msg)

    def call(self, method, **params):
        self.n += 1
        data = json.dumps({"id": self.n, "method": method, "params": params}).encode()
        mask = os.urandom(4)
        head = bytes([0x81])
        head += bytes([0x80 | len(data)]) if len(data) < 126 else (
            bytes([0x80 | 126]) + struct.pack(">H", len(data)) if len(data) < 65536
            else bytes([0x80 | 127]) + struct.pack(">Q", len(data)))
        self.s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))
        while True:
            m = self._recv()
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    def shot(self, path):
        data = self.call("Page.captureScreenshot", format="png")["data"]
        Path(path).write_bytes(base64.b64decode(data))


def wait_for(cdp, expr, timeout=40):
    end = time.time() + timeout
    while time.time() < end:
        if cdp.js(expr):
            return
        time.sleep(0.2)
    raise TimeoutError(expr)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def record_gesture(port, feeder, name, pose, action, takes=4):
    """The add flow: name -> takes -> check -> save (the same calls the UI makes)."""
    run = asyncio.get_running_loop().run_in_executor
    await run(None, http, port, "POST", "/api/candidate", {"name": name})
    feeder.pose = pose
    for i in range(takes):
        feeder.seed = i
        await run(None, http, port, "POST", "/api/candidate/record",
                  {"kind": "take", "countdown_s": 0, "seconds": 0.8})
        await asyncio.sleep(0.3)
    report = await run(None, http, port, "POST", "/api/candidate/assess")
    print(f"  {name}: {report['verdict']}")
    await run(None, http, port, "POST", "/api/candidate/save", {"action": action})


async def main():
    tmp = Path(tempfile.mkdtemp(prefix="jarvis-shots-"))
    lib = str(tmp / "gestures.json")       # never ~/.jarvis
    os.environ.pop("JARVIS_STUDIO_PORT", None)
    bus = EventBus()
    studio = StudioModule("studio", {"port": free_port(), "library_path": lib})
    await studio.setup(bus)
    studio.runner.backend = FakeBackend()
    hci = GestureHCIModule("gesture_hci", {"gestures": {"library_path": lib}})
    await hci.setup(bus)
    await studio.server.start()
    port = studio.server.port
    feeder = Feeder(bus)
    feed = asyncio.create_task(feeder.run())
    chrome = None
    try:
        for name, pose, action in GESTURES:
            await record_gesture(port, feeder, name, pose, action)
        feeder.pose = PEACE
        await asyncio.sleep(1.5)   # let the engine pick up the saved gestures and activate peace

        def browse():
            nonlocal chrome
            dbg = free_port()
            chrome = subprocess.Popen(
                [CHROME, "--headless=new", f"--remote-debugging-port={dbg}", "--hide-scrollbars",
                 f"--user-data-dir={tmp / 'chrome'}", "--no-first-run", f"--window-size={W},{H}",
                 "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(100):
                try:
                    tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{dbg}/json/list"))
                    page = next(t for t in tabs if t["type"] == "page")
                    break
                except Exception:
                    time.sleep(0.1)
            cdp = CDP(page["webSocketDebuggerUrl"])
            cdp.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
            cdp.call("Page.navigate", url=f"http://127.0.0.1:{port}/")
            wait_for(cdp, "document.querySelectorAll('#gesture-list li').length >= 6")
            wait_for(cdp, "document.getElementById('active').textContent.trim().toLowerCase() === 'peace'")
            time.sleep(1)
            cdp.shot(OUT / "studio.png")
            # The add flow, clicked through the real UI: name, three takes, check.
            cdp.js("document.getElementById('add-btn').click()")
            cdp.js("document.getElementById('name-input').value='four';"
                   "document.getElementById('name-form').requestSubmit()")
            wait_for(cdp, "add.name === 'four'")
            return cdp

        loop = asyncio.get_running_loop()
        cdp = await loop.run_in_executor(None, browse)

        def add_flow():
            feeder.pose = FOUR
            for i in range(3):
                feeder.seed = 10 + i
                cdp.js("document.getElementById('take-btn').click()")
                wait_for(cdp, f"add.takes.length === {i + 1} && !add.busy", 30)
                time.sleep(0.4)
            cdp.js("document.getElementById('check-btn').click()")
            wait_for(cdp, "add.report !== null && !add.busy")
            # Tall enough that the whole flow fits without scrolling.
            cdp.call("Emulation.setDeviceMetricsOverride", width=W, height=1300, deviceScaleFactor=2, mobile=False)
            cdp.js("window.scrollTo(0, 0)")
            time.sleep(1.0)
            cdp.shot(OUT / "studio-add.png")

        await loop.run_in_executor(None, add_flow)
    finally:
        feed.cancel()
        if chrome:
            chrome.terminate()
        await studio.server.stop()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    asyncio.run(main())
    for p in sorted(OUT.glob("studio*.png")):
        print(p, p.stat().st_size // 1024, "KB")
