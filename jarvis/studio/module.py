"""The gesture studio: a local web UI served by the running Jarvis process
(so it shares the camera). See it at http://127.0.0.1:8765.

- Live: the tracked hand, the active gesture, per-gesture match scores, the
  fist-scroll tilt.
- Gestures: built-ins and custom gestures, their quality, their bindings.
- Add: name -> record takes -> optional everyday-movement clip -> quality
  report -> bind -> save. While a take records, cursor control and gesture
  actions pause (StudioModeEvent).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from ..events import GestureEvent, GestureStateEvent, StudioModeEvent, WorldPoseEvent
from ..gestures.actions import MEDIA_KEYS, TYPES, ActionRunner, describe, normalize, validate
from ..gestures.builtins import BUILTINS, RESERVED
from ..gestures.library import GestureLibrary, check_name
from ..gestures.quality import assess
from ..runtime.module_base import Module
from .http import HttpServer, Request, Response, StudioError
from .recorder import to_frames

logger = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
_CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                  ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}
_BUILTIN_INFO = {
    "pinch": "Thumb to index: quick = click, hold = drag.",
    "fist": "Close your hand, then tilt: scroll.",
    "point": "Index out, others curled.",
}
MAX_TAKES = 8


class StudioModule(Module):
    async def setup(self, bus) -> None:
        await super().setup(bus)
        self.library = GestureLibrary(self.config.get("library_path"))
        self.runner = ActionRunner()
        # JARVIS_STUDIO_PORT wins over the config, so run_studio.sh can step
        # around a port something else already holds.
        port = int(os.environ.get("JARVIS_STUDIO_PORT") or self.config.get("port", 8765))
        self.server = HttpServer(self.config.get("host", "127.0.0.1"), port)
        self._routes()
        self._hand: Optional[Tuple[float, np.ndarray]] = None
        self._state: Optional[GestureStateEvent] = None
        self._events: Deque[Dict[str, Any]] = deque(maxlen=12)
        self._recording: Optional[Dict[str, Any]] = None   # {kind, phase, until, samples}
        self.candidate: Optional[Dict[str, Any]] = None
        bus.subscribe(WorldPoseEvent, self._on_pose)
        bus.subscribe(GestureStateEvent, self._on_state)
        bus.subscribe(GestureEvent, self._on_gesture)

    # --- bus ---

    async def _on_pose(self, event: WorldPoseEvent) -> None:
        if event.hand_shape is None:
            return
        self._hand = (event.timestamp, event.hand_shape)
        rec = self._recording
        if rec and rec["phase"] == "recording":
            rec["samples"].append((event.timestamp, np.array(event.hand_shape, dtype=np.float32)))

    async def _on_state(self, event: GestureStateEvent) -> None:
        self._state = event

    async def _on_gesture(self, event: GestureEvent) -> None:
        self._events.append({"name": event.name, "phase": event.phase, "t": time.time()})

    # --- views ---

    def _gesture_entry(self, name: str) -> Dict[str, Any]:
        self.library.reload_if_changed()
        custom = self.library.gestures.get(name)
        binding = self.library.bindings.get(name)
        if binding is None and name == "point":
            binding = {"type": "keys", "combo": "cmd+space"}       # dispatcher default
        q = (custom or {}).get("quality") or {}
        return {
            "name": name,
            "kind": "custom" if custom else "builtin",
            "reserved": name in RESERVED,
            "info": _BUILTIN_INFO.get(name, ""),
            "binding": binding,
            "binding_text": "Click / drag" if name == "pinch" else "Scroll" if name == "fist"
                            else describe(binding),
            "verdict": q.get("verdict"),
            "checks": {k: {"verdict": c.get("verdict"), "score": c.get("score")}
                       for k, c in (q.get("checks") or {}).items()},
            "created": (custom or {}).get("created"),
        }

    def _snapshot(self) -> Dict[str, Any]:
        now = time.perf_counter()
        hand = None
        if self._hand and now - self._hand[0] < 0.3:
            hand = np.round(self._hand[1], 4).tolist()
        st = self._state
        rec = self._recording
        return {
            "hand": hand,
            "active": st.active if st else None,
            "candidate": st.candidate if st else None,
            "scores": {k: round(v, 3) for k, v in (st.scores if st else {}).items()},
            "pinch": st.pinch if st else None,
            "scroll": st.scroll if st else None,
            "recording": None if not rec else {
                "kind": rec["kind"], "phase": rec["phase"],
                "remaining": round(max(0.0, rec["until"] - now), 2),
                "frames": len(rec["samples"]),
            },
            "events": list(self._events),
            "candidate_takes": len(self.candidate["takes"]) if self.candidate else 0,
        }

    # --- routes ---

    def _routes(self) -> None:
        r = self.server.route

        @r("GET", r"/")
        async def index(req: Request):
            return self._static("index.html")

        @r("GET", r"/static/(?P<name>[A-Za-z0-9_.-]+)")
        async def static(req: Request, name: str):
            return self._static(name)

        @r("GET", r"/api/meta")
        async def meta(req: Request):
            return {"media_keys": MEDIA_KEYS, "action_types": list(TYPES),
                    "builtins": list(BUILTINS), "reserved": list(RESERVED), "max_takes": MAX_TAKES}

        @r("GET", r"/api/state")
        async def state(req: Request):
            return self._snapshot()

        @r("GET", r"/api/stream")
        async def stream(req: Request):
            async def events():
                while True:
                    yield self._snapshot()
                    await asyncio.sleep(1 / 20)
            return events()

        @r("GET", r"/api/gestures")
        async def gestures(req: Request):
            self.library.reload_if_changed()
            names = list(BUILTINS) + sorted(self.library.gestures)
            return {"gestures": [self._gesture_entry(n) for n in names]}

        @r("DELETE", r"/api/gestures/(?P<name>[^/]+)")
        async def delete(req: Request, name: str):
            name = _unquote(name)
            self.library.reload_if_changed()
            if name not in self.library.gestures:
                raise StudioError(f"no custom gesture called '{name}'")
            self.library.remove_gesture(name)
            return {"ok": True}

        @r("POST", r"/api/gestures/(?P<name>[^/]+)/binding")
        async def bind(req: Request, name: str):
            name = _unquote(name)
            self.library.reload_if_changed()
            try:
                self.library.set_binding(name, (req.json or {}).get("action"))
            except ValueError as e:
                raise StudioError(str(e))
            return self._gesture_entry(name)

        @r("POST", r"/api/actions/validate")
        async def check_action(req: Request):
            action = (req.json or {}).get("action")
            return {"errors": validate(action), "text": describe(action)}

        @r("POST", r"/api/actions/test")
        async def test_action(req: Request):
            action = normalize((req.json or {}).get("action"))
            errors = validate(action)
            if errors:
                raise StudioError("; ".join(errors))
            # Give the browser a moment to lose focus, then run off the loop.
            await asyncio.sleep(0.3)
            ok = await asyncio.get_running_loop().run_in_executor(None, self.runner.run, action)
            return {"ok": ok}

        @r("POST", r"/api/candidate")
        async def new_candidate(req: Request):
            body = req.json or {}
            name = str(body.get("name", "")).strip()
            replace = bool(body.get("replace"))
            self.library.reload_if_changed()
            if replace:
                if name not in self.library.gestures:
                    raise StudioError(f"no custom gesture called '{name}' to retrain")
            else:
                problems = check_name(name)
                if name in self.library.gestures:
                    problems.append(f"'{name}' already exists (retrain it from its card)")
                if problems:
                    raise StudioError(problems[0])
            self.candidate = {"name": name, "replace": replace, "takes": [], "background": None,
                              "report": None}
            return {"name": name, "replace": replace}

        @r("POST", r"/api/candidate/record")
        async def record(req: Request):
            cand = self._need_candidate()
            body = req.json or {}
            kind = body.get("kind", "take")
            if kind not in ("take", "background"):
                raise StudioError("kind must be take or background")
            if kind == "take" and len(cand["takes"]) >= MAX_TAKES:
                raise StudioError(f"{MAX_TAKES} takes is plenty; delete one to redo it")
            countdown = float(np.clip(body.get("countdown_s", 3.0), 0.0, 10.0))
            default_s = 1.5 if kind == "take" else 10.0
            seconds = float(np.clip(body.get("seconds", default_s), 0.2, 30.0))
            frames = await self._record(kind, countdown, seconds)
            tracked = sum(f is not None for f in frames) / max(len(frames), 1)
            if kind == "take":
                cand["takes"].append(frames)
            else:
                cand["background"] = frames
            cand["report"] = None
            return {"kind": kind, "frames": len(frames), "tracked": round(tracked, 3),
                    "takes": len(cand["takes"]), "has_background": cand["background"] is not None}

        @r("DELETE", r"/api/candidate/takes/(?P<idx>\d+)")
        async def delete_take(req: Request, idx: str):
            cand = self._need_candidate()
            i = int(idx) - 1
            if not 0 <= i < len(cand["takes"]):
                raise StudioError(f"there's no take {idx}")
            cand["takes"].pop(i)
            cand["report"] = None
            return {"takes": len(cand["takes"])}

        @r("DELETE", r"/api/candidate/background")
        async def delete_background(req: Request):
            cand = self._need_candidate()
            cand["background"] = None
            cand["report"] = None
            return {"ok": True}

        @r("POST", r"/api/candidate/assess")
        async def assess_candidate(req: Request):
            cand = self._need_candidate()
            if not cand["takes"]:
                raise StudioError("record some takes first")
            report = await asyncio.get_running_loop().run_in_executor(None, self._assess, cand)
            cand["report"] = report
            return {k: v for k, v in report.items() if k != "samples"}

        @r("POST", r"/api/candidate/save")
        async def save(req: Request):
            cand = self._need_candidate()
            report = cand.get("report")
            if report is None:
                raise StudioError("run the check first")
            if report["verdict"] == "fail":
                raise StudioError("this gesture failed the check; fix what it flagged and re-check")
            name = cand["name"]
            self.library.reload_if_changed()
            old_binding = self.library.bindings.get(name) if cand["replace"] else None
            quality = {k: v for k, v in report.items() if k not in ("samples",)}
            self.library.add_gesture(name, np.asarray(report["samples"], dtype=np.float32),
                                     report["radius"], quality)
            action = (req.json or {}).get("action", old_binding)
            if action:
                try:
                    self.library.set_binding(name, action)
                except ValueError as e:
                    raise StudioError(f"saved, but the binding was rejected: {e}")
            self.candidate = None
            return self._gesture_entry(name)

    def _static(self, name: str) -> Response:
        path = STATIC / name
        if not path.is_file() or path.parent != STATIC:
            return Response(404, b'{"error": "not found"}')
        return Response(200, path.read_bytes(),
                        _CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

    def _need_candidate(self) -> Dict[str, Any]:
        if not self.candidate:
            raise StudioError("start a new gesture first")
        return self.candidate

    def _assess(self, cand: Dict[str, Any]) -> Dict[str, Any]:
        self.library.reload_if_changed()
        # Retraining: don't compare the gesture with its own old version.
        others = {n: g for n, g in self.library.gestures.items() if n != cand["name"]}
        return assess(cand["takes"], SimpleNamespace(gestures=others), background=cand["background"])

    async def _record(self, kind: str, countdown: float, seconds: float) -> List[Optional[np.ndarray]]:
        if self._recording:
            raise StudioError("already recording")
        now = time.perf_counter()
        rec = {"kind": kind, "phase": "countdown", "until": now + countdown, "samples": []}
        self._recording = rec
        # Cursor and actions pause from the countdown on: the hand is busy
        # getting into the gesture, and that must not click or fire anything.
        await self.bus.publish(StudioModeEvent(recording=True))
        try:
            await asyncio.sleep(countdown)
            start = time.perf_counter()
            rec.update(phase="recording", until=start + seconds)
            await asyncio.sleep(seconds)
            end = time.perf_counter()
            samples = [(t, s) for t, s in rec["samples"] if start <= t <= end]
            return to_frames(samples, start, end)
        finally:
            self._recording = None
            await self.bus.publish(StudioModeEvent(recording=False))

    # --- lifecycle ---

    async def run(self) -> None:
        await self.server.start()
        logger.info("Gesture studio: http://%s:%d", self.server.host, self.server.port)
        while True:
            await asyncio.sleep(3600)

    async def teardown(self) -> None:
        await self.server.stop()


def _unquote(s: str) -> str:
    from urllib.parse import unquote
    return unquote(s)
