"""Wake-word detection: prebuilt openWakeWord models plus drop-in *.onnx files."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openwakeword
from openwakeword.model import Model


def _prebuilt_path(name: str) -> Path:
    """Locate a prebuilt onnx model by name, downloading models if missing."""
    def find() -> Path | None:
        for p in openwakeword.get_pretrained_model_paths("onnx"):
            if re.sub(r"_v\d+(\.\d+)*$", "", Path(p).stem) == name and Path(p).exists():
                return Path(p)
        return None

    path = find()
    if path is None:
        openwakeword.utils.download_models()
        path = find()
    if path is None:
        raise FileNotFoundError(f"prebuilt wake word model not found: {name}")
    return path


class WakeWord:
    """Scores 1280-sample int16 frames; `process` returns the model name that fired."""

    def __init__(self, models_dir: Path, threshold: float,
                 prebuilt: list[str] | None = None) -> None:
        self._threshold = threshold
        paths: dict[str, Path] = {}
        for name in (["hey_jarvis"] if prebuilt is None else prebuilt):
            paths[name] = _prebuilt_path(name)
        models_dir = Path(models_dir)
        if models_dir.is_dir():
            for p in sorted(models_dir.glob("*.onnx")):
                paths[p.stem] = p
        self._paths = paths
        self.names: list[str] = list(paths)
        self._model = self._build()

    def _build(self) -> Model:
        return Model(
            wakeword_models=[str(p) for p in self._paths.values()],
            inference_framework="onnx",
        )

    def _key_to_name(self, key: str) -> str:
        for name, p in self._paths.items():
            if p.stem == key:
                return name
        return key

    def process(self, frame: np.ndarray) -> str | None:
        scores = self._model.predict(frame)
        best: tuple[float, str] | None = None
        for key, score in scores.items():
            if score >= self._threshold and (best is None or score > best[0]):
                best = (float(score), self._key_to_name(key))
        return best[1] if best else None

    def reset(self) -> None:
        # Model.reset() does not clear the audio/feature buffer; rebuild instead.
        self._model = self._build()
