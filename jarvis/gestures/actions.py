"""What a gesture does: media keys, key combos, typed text, shell commands,
and macros (ordered steps of those plus waits).

An action is a plain dict so it round-trips through gestures.json and the
studio UI unchanged:

    {"type": "media", "key": "play_pause"}
    {"type": "keys", "combo": "cmd+shift+4"}
    {"type": "text", "text": "on my way"}
    {"type": "shell", "command": "jarvis-dj next"}
    {"type": "macro", "steps": [{"type": "keys", "combo": "cmd+c"},
                                {"type": "wait", "seconds": 0.2}, ...]}
    {"type": "none"}            # explicitly do nothing (turns a default off)

Any action may carry "repeat_s": while the gesture is held, fire again every
repeat_s seconds (volume up held = keeps turning up).
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MEDIA_KEYS = {
    "play_pause": "play / pause",
    "next": "next track",
    "prev": "previous track",
    "volume_up": "volume up",
    "volume_down": "volume down",
    "mute": "mute",
}

_MODIFIERS = {
    "cmd": "command", "command": "command",
    "ctrl": "ctrl", "control": "ctrl",
    "alt": "option", "opt": "option", "option": "option",
    "shift": "shift",
}
_NAMED = {
    "space": "space", "enter": "enter", "return": "enter", "tab": "tab",
    "esc": "esc", "escape": "esc", "up": "up", "down": "down", "left": "left",
    "right": "right", "delete": "backspace", "backspace": "backspace",
    "forwarddelete": "delete", "home": "home", "end": "end",
    "pageup": "pageup", "pagedown": "pagedown",
    **{f"f{i}": f"f{i}" for i in range(1, 21)},
}
_PUNCT = set("`-=[]\\;',./")

# Names the dispatcher accepted before actions were specs.
_LEGACY = {
    "raycast": {"type": "keys", "combo": "cmd+space"},
    "media_next": {"type": "media", "key": "next"},
    "media_prev": {"type": "media", "key": "prev"},
    "media_play_pause": {"type": "media", "key": "play_pause"},
    "volume_up": {"type": "media", "key": "volume_up"},
    "volume_down": {"type": "media", "key": "volume_down"},
}

TYPES = ("media", "keys", "text", "shell", "macro", "none")
MIN_REPEAT_S = 0.05


def parse_combo(combo: str) -> List[str]:
    """"cmd+shift+4" -> ["command", "shift", "4"] (pyautogui key names).
    Raises ValueError with a plain-words reason."""
    parts = [p.strip().lower() for p in (combo or "").split("+")]
    if not combo or any(p == "" for p in parts):
        raise ValueError(f"'{combo}' isn't a key combo (try cmd+shift+4)")
    *mods, key = parts
    keys = []
    for m in mods:
        if m not in _MODIFIERS:
            raise ValueError(f"'{m}' isn't a modifier (cmd, ctrl, option, shift)")
        keys.append(_MODIFIERS[m])
    if key in _MODIFIERS:
        raise ValueError(f"'{combo}' has no key after the modifiers")
    if key in _NAMED:
        keys.append(_NAMED[key])
    elif len(key) == 1 and (key.isalnum() or key in _PUNCT):
        keys.append(key)
    else:
        raise ValueError(f"'{key}' isn't a key name")
    return keys


def normalize(action: Any) -> Optional[Dict[str, Any]]:
    """Legacy name or spec -> spec. ""/None -> None (no action)."""
    if not action:
        return None
    if isinstance(action, str):
        return dict(_LEGACY[action]) if action in _LEGACY else {"type": "legacy", "name": action}
    return dict(action)


def validate(action: Any, _depth: int = 0) -> List[str]:
    """Problems with an action, in plain words. [] = fine."""
    spec = normalize(action)
    if spec is None:
        return []
    t = spec.get("type")
    errors: List[str] = []
    rep = spec.get("repeat_s")
    if rep is not None and (not isinstance(rep, (int, float)) or rep < MIN_REPEAT_S):
        errors.append(f"repeat must be at least {MIN_REPEAT_S} s")
    if t == "none" and _depth == 0:
        pass
    elif t == "legacy":
        errors.append(f"unknown action '{spec.get('name')}'")
    elif t == "media":
        if spec.get("key") not in MEDIA_KEYS:
            errors.append(f"media key must be one of: {', '.join(MEDIA_KEYS)}")
    elif t == "keys":
        try:
            parse_combo(spec.get("combo", ""))
        except ValueError as e:
            errors.append(str(e))
    elif t == "text":
        if not spec.get("text"):
            errors.append("text to type is empty")
    elif t == "shell":
        if not str(spec.get("command", "")).strip():
            errors.append("shell command is empty")
    elif t == "wait" and _depth > 0:
        s = spec.get("seconds")
        if not isinstance(s, (int, float)) or not 0 <= s <= 10:
            errors.append("wait must be 0-10 seconds")
    elif t == "macro" and _depth == 0:
        steps = spec.get("steps") or []
        if not steps:
            errors.append("macro has no steps")
        for i, step in enumerate(steps, 1):
            errors += [f"step {i}: {e}" for e in validate(step, _depth + 1)]
    elif t == "macro":
        errors.append("macros can't contain macros")
    else:
        errors.append(f"unknown action type '{t}'")
    return errors


def describe(action: Any) -> str:
    spec = normalize(action)
    if spec is None:
        return "Nothing"
    t = spec.get("type")
    if t == "none":
        return "Nothing"
    if t == "media":
        return f"Media: {MEDIA_KEYS.get(spec.get('key'), spec.get('key'))}"
    if t == "keys":
        return f"Keys: {spec.get('combo')}"
    if t == "text":
        return f"Type: {spec.get('text')}"
    if t == "shell":
        return f"Shell: {spec.get('command')}"
    if t == "wait":
        return f"Wait {spec.get('seconds')} s"
    if t == "macro":
        return f"Macro: {len(spec.get('steps') or [])} steps"
    return f"Unknown ({t})"


class MacBackend:
    """Real key / media / shell output."""

    def media(self, key: str) -> None:
        from pynput.keyboard import Controller, Key
        k = {"play_pause": Key.media_play_pause, "next": Key.media_next,
             "prev": Key.media_previous, "volume_up": Key.media_volume_up,
             "volume_down": Key.media_volume_down, "mute": Key.media_volume_mute}[key]
        kb = Controller()
        kb.press(k)
        kb.release(k)

    def hotkey(self, keys: List[str]) -> None:
        import pyautogui
        pyautogui.hotkey(*keys)

    def text(self, s: str) -> None:
        import pyautogui
        pyautogui.write(s)

    def shell(self, cmd: str) -> None:
        # Fire and forget; the command outlives a slow gesture loop.
        subprocess.Popen(cmd, shell=True, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def sleep(self, s: float) -> None:
        time.sleep(s)


class ActionRunner:
    def __init__(self, backend: Any = None) -> None:
        self.backend = backend or MacBackend()

    def run(self, action: Any) -> bool:
        """Run an action now (blocking through macro waits). False if it was
        invalid or failed; never raises."""
        spec = normalize(action)
        if spec is None or spec.get("type") == "none":
            return False
        errors = validate(spec)
        if errors:
            logger.warning("Not running %s: %s", describe(spec), "; ".join(errors))
            return False
        try:
            self._run(spec)
            return True
        except Exception:
            logger.exception("Action failed: %s", describe(spec))
            return False

    def _run(self, spec: Dict[str, Any]) -> None:
        t, b = spec["type"], self.backend
        if t == "media":
            b.media(spec["key"])
        elif t == "keys":
            b.hotkey(parse_combo(spec["combo"]))
        elif t == "text":
            b.text(spec["text"])
        elif t == "shell":
            b.shell(spec["command"])
        elif t == "wait":
            b.sleep(float(spec["seconds"]))
        elif t == "macro":
            for step in spec["steps"]:
                self._run(normalize(step))

    @staticmethod
    def is_slow(action: Any) -> bool:
        """Macros with waits block; run those off the event loop."""
        spec = normalize(action) or {}
        return spec.get("type") == "macro"
