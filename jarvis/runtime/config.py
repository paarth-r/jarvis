from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import yaml


@dataclass
class CameraConfig:
    device: int
    width: int
    height: int
    fps: int
    intrinsics: Dict[str, float]
    extrinsics: Dict[str, Any]
    source: str = "device"   # "device" (live) | "video"/"file"/"images" (replay)
    path: str = ""           # file or image-directory path when source != device


@dataclass
class ModuleConfig:
    class_path: str
    enabled: bool
    settings: Dict[str, Any]


@dataclass
class SubprocessConfig:
    enabled: bool
    module: str


@dataclass
class JarvisConfig:
    cameras: Dict[str, CameraConfig]
    modules: Dict[str, ModuleConfig]
    subprocesses: Dict[str, SubprocessConfig]
    hand_pose: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str) -> JarvisConfig:
    with open(path) as f:          # raises FileNotFoundError if missing
        raw = yaml.safe_load(f)

    cameras: Dict[str, CameraConfig] = {}
    for name, cam in (raw.get("cameras") or {}).items():
        cameras[name] = CameraConfig(
            device=cam.get("device", 0),
            width=cam.get("width", 1280),
            height=cam.get("height", 800),
            fps=cam.get("fps", 30),
            intrinsics=cam.get("intrinsics", {}),
            extrinsics=cam.get("extrinsics", {}),
            source=cam.get("source", "device"),
            path=cam.get("path", ""),
        )

    modules: Dict[str, ModuleConfig] = {}
    for name, mod in (raw.get("modules") or {}).items():
        modules[name] = ModuleConfig(
            class_path=mod["class"],
            enabled=mod.get("enabled", True),
            settings={k: v for k, v in mod.items() if k not in ("class", "enabled")},
        )

    subprocesses: Dict[str, SubprocessConfig] = {}
    for name, sub in (raw.get("subprocesses") or {}).items():
        subprocesses[name] = SubprocessConfig(
            enabled=sub.get("enabled", False),
            module=sub.get("module", ""),
        )

    hand_pose: Dict[str, Any] = dict(raw.get("hand_pose") or {})

    return JarvisConfig(
        cameras=cameras, modules=modules, subprocesses=subprocesses,
        hand_pose=hand_pose,
    )
