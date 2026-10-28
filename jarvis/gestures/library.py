"""~/.jarvis/gestures.json: custom gestures (feature samples + acceptance
radius + the quality report they were saved with) and gesture -> action
bindings. The runtime and the studio both read it; writes are atomic and
readers reload when the file's mtime changes."""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .actions import normalize, validate
from .builtins import BUILTINS, RESERVED

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path.home() / ".jarvis" / "gestures.json"
_NAME = re.compile(r"^[a-z0-9][a-z0-9 _-]{0,31}$")


def check_name(name: str) -> List[str]:
    """Problems with a new gesture's name, in plain words. [] = fine."""
    if not name or not name.strip():
        return ["give it a name"]
    if name.lower() in BUILTINS:
        return [f"'{name}' is a built-in gesture"]
    if not _NAME.match(name):
        return ["use 1-32 lowercase letters, digits, spaces, - or _"]
    return []


class GestureLibrary:
    def __init__(self, path: Optional[os.PathLike] = None) -> None:
        # JARVIS_GESTURES_PATH lets tests (and experiments) use another file.
        self.path = Path(path or os.environ.get("JARVIS_GESTURES_PATH") or DEFAULT_PATH).expanduser()
        self.gestures: Dict[str, Dict[str, Any]] = {}
        self.bindings: Dict[str, Dict[str, Any]] = {}
        self._mtime: Optional[float] = None
        self.load()

    # --- persistence ---

    def _stat(self) -> Optional[float]:
        try:
            return self.path.stat().st_mtime
        except FileNotFoundError:
            return None

    def load(self) -> None:
        self._mtime = self._stat()
        if self._mtime is None:
            self.gestures, self.bindings = {}, {}
            return
        try:
            data = json.loads(self.path.read_text())
            self.gestures = dict(data.get("gestures") or {})
            self.bindings = dict(data.get("bindings") or {})
        except (ValueError, OSError):
            aside = self.path.with_name(self.path.name + ".corrupt")
            logger.error("Unreadable %s; moved to %s and starting empty", self.path, aside)
            os.replace(self.path, aside)
            self.gestures, self.bindings, self._mtime = {}, {}, None

    def reload_if_changed(self) -> bool:
        if self._stat() == self._mtime:
            return False
        self.load()
        return True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "gestures": self.gestures, "bindings": self.bindings}
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".gestures-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._mtime = self._stat()

    # --- gestures ---

    def samples(self, name: str) -> np.ndarray:
        return np.asarray(self.gestures[name]["samples"], dtype=np.float32)

    def add_gesture(self, name: str, samples: np.ndarray, radius: float,
                    quality: Optional[dict] = None) -> None:
        self.gestures[name] = {
            "samples": np.round(np.asarray(samples, dtype=float), 5).tolist(),
            "radius": float(radius),
            "quality": quality or {},
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.save()

    def remove_gesture(self, name: str) -> None:
        self.gestures.pop(name, None)
        self.bindings.pop(name, None)
        self.save()

    # --- bindings ---

    def set_binding(self, name: str, action: Any) -> None:
        """Bind (or with None, unbind) a gesture. Raises ValueError with a
        plain-words reason."""
        if name in RESERVED:
            raise ValueError(f"'{name}' is reserved ({'click / drag' if name == 'pinch' else 'scroll'})")
        if name not in BUILTINS and name not in self.gestures:
            raise ValueError(f"no gesture called '{name}'")
        spec = normalize(action)
        if spec is None:
            self.bindings.pop(name, None)
        else:
            errors = validate(spec)
            if errors:
                raise ValueError("; ".join(errors))
            self.bindings[name] = spec
        self.save()
