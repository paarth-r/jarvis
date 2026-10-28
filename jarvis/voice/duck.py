"""Lower other apps' own volume while Jarvis is engaged, then put it back.

Only the apps' internal `sound volume` is touched, never the system volume Jarvis
speaks at. Pre-duck levels are persisted so a crash cannot leave music stuck low.
All methods are synchronous; async callers use asyncio.to_thread.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from jarvis.voice.mac_speakers import Osascript, run_osascript

log = logging.getLogger(__name__)

FADE_STEPS = 4
FADE_PAUSE = 0.12


class MusicDucker:
    def __init__(self, osascript: Osascript = run_osascript,
                 apps: tuple[str, ...] = ("Spotify", "Music"), ratio: float = 0.2,
                 state_path: Path | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self._osa = osascript
        self._apps = apps
        self._ratio = ratio
        self._state_path = state_path
        self._sleep = sleep
        self._ducked: dict[str, tuple[int, int]] = {}  # app -> (previous, set_to)
        self._lock = threading.Lock()

    def duck(self) -> None:
        with self._lock:
            for app in self._apps:
                if app in self._ducked:
                    continue
                try:
                    if not self._running(app):
                        continue
                    volume = int(self._osa(f'tell application "{app}" to sound volume'))
                    target = max(1, round(volume * self._ratio))
                    if volume <= 0 or target >= volume:
                        continue
                    # recorded before the set so a crash in between cannot strand the music low
                    self._ducked[app] = (volume, target)
                    self._save()
                    self._osa(f'tell application "{app}" to set sound volume to {target}')
                except Exception as exc:
                    log.warning("could not duck %s: %s", app, exc)
                    if self._ducked.pop(app, None) is not None:
                        self._save()

    def restore(self) -> None:
        with self._lock:
            self._restore()

    def recover(self) -> None:
        """Undo a ducking left behind by a crashed run."""
        if self._state_path is None:
            return
        with self._lock:
            if not self._state_path.exists():
                return
            try:
                raw = json.loads(self._state_path.read_text())
                leftover = {app: (int(prev), int(set_to)) for app, (prev, set_to) in raw.items()}
            except Exception as exc:
                log.warning("ignoring corrupt duck state %s: %s", self._state_path, exc)
                self._forget_state()
                return
            for app, levels in leftover.items():
                self._ducked.setdefault(app, levels)
            self._restore()

    def _running(self, app: str) -> bool:
        return self._osa(f'application "{app}" is running') == "true"

    def _restore(self) -> None:
        for app, (previous, set_to) in list(self._ducked.items()):
            try:
                if not self._running(app):
                    log.debug("%s quit while ducked", app)
                    del self._ducked[app]
                    continue
                current = int(self._osa(f'tell application "{app}" to sound volume'))
                ours = {set_to + round((previous - set_to) * s / FADE_STEPS)
                        for s in range(FADE_STEPS + 1)}  # set_to, or left mid-fade by a failed restore
                if current in ours:
                    for step in range(1, FADE_STEPS + 1):
                        level = current + round((previous - current) * step / FADE_STEPS)
                        self._osa(f'tell application "{app}" to set sound volume to {level}')
                        if step < FADE_STEPS:
                            self._sleep(FADE_PAUSE)
                # else the user changed it while ducked; keep their choice
                del self._ducked[app]
            except Exception as exc:
                log.warning("could not restore %s (will retry): %s", app, exc)
        self._save()

    def _save(self) -> None:
        if self._state_path is None:
            return
        if not self._ducked:
            self._forget_state()
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(self._ducked))
        except OSError as exc:
            log.warning("could not write duck state: %s", exc)

    def _forget_state(self) -> None:
        if self._state_path is None:
            return
        try:
            self._state_path.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not delete duck state: %s", exc)
