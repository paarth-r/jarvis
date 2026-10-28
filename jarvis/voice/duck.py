"""Lower other apps' own volume while Jarvis is engaged, then put it back.

Only the apps' internal `sound volume` is touched, never the system volume Jarvis
speaks at. Pre-duck levels are persisted so a crash cannot leave music stuck low.
All methods are synchronous; async callers use asyncio.to_thread.
"""
from __future__ import annotations

import json
import logging
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

    def duck(self) -> None:
        changed = False
        for app in self._apps:
            if app in self._ducked:
                continue
            try:
                if self._osa(f'application "{app}" is running') != "true":
                    continue
                volume = int(self._osa(f'sound volume of application "{app}"'))
                if volume <= 0:
                    continue
                target = max(1, round(volume * self._ratio))
                if target >= volume:
                    continue
                self._osa(f'set sound volume of application "{app}" to {target}')
            except Exception as exc:
                log.warning("could not duck %s: %s", app, exc)
                continue
            self._ducked[app] = (volume, target)
            changed = True
        if changed:
            self._save()

    def restore(self) -> None:
        self._restore(self._ducked)
        self._ducked = {}
        self._forget_state()

    def recover(self) -> None:
        """Undo a ducking left behind by a crashed run."""
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text())
            leftover = {app: (int(prev), int(set_to)) for app, (prev, set_to) in raw.items()}
        except Exception as exc:
            log.warning("ignoring corrupt duck state %s: %s", self._state_path, exc)
            self._forget_state()
            return
        self._restore(leftover)
        self._forget_state()

    def _restore(self, ducked: dict[str, tuple[int, int]]) -> None:
        for app, (previous, set_to) in ducked.items():
            try:
                current = int(self._osa(f'sound volume of application "{app}"'))
                if current != set_to:
                    continue  # the user changed it while ducked; keep their choice
                for step in range(1, FADE_STEPS + 1):
                    level = current + round((previous - current) * step / FADE_STEPS)
                    self._osa(f'set sound volume of application "{app}" to {level}')
                    if step < FADE_STEPS:
                        self._sleep(FADE_PAUSE)
            except Exception as exc:
                log.warning("could not restore %s: %s", app, exc)

    def _save(self) -> None:
        if self._state_path is None:
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
