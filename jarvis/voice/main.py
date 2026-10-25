"""`jarvis-voice` entry point: build the real dependencies and run the voice loop."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from jarvis.voice.config import PATH_FIELDS, VoiceConfig, load_config

DEFAULT_CONFIG = Path("config.voice.example.yaml")
DEFAULT_LOCAL = Path("config.voice.local.yaml")
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_config_path(path: Path, root: Path = REPO_ROOT) -> Path:
    """A relative path missing from the cwd falls back to the repo root.

    Lets a manual run from another directory find the repo's config and models.
    """
    if path.is_absolute() or path.exists():
        return path
    candidate = root / path
    return candidate if candidate.exists() else path


def resolve_paths(cfg: VoiceConfig, root: Path = REPO_ROOT) -> VoiceConfig:
    """Apply the repo-root fallback to every relative path in the config."""
    changes = {
        name: resolve_config_path(value, root)
        for name in PATH_FIELDS
        if isinstance(value := getattr(cfg, name), Path)
    }
    return replace(cfg, **changes)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="jarvis-voice",
        description="Always-on 'hey Jarvis' voice front-end for headless Claude Code.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help="base YAML config (default: %(default)s)")
    p.add_argument("--local", type=Path, default=DEFAULT_LOCAL,
                   help="local override YAML, applied if it exists (default: %(default)s)")
    p.add_argument("--list-devices", action="store_true",
                   help="print audio input/output devices and exit")
    p.add_argument("--no-room", action="store_true",
                   help="do not register the room MCP server")
    p.add_argument("--check", action="store_true",
                   help="construct every dependency (models, room config, brain), "
                        "print 'ok' and exit without listening")
    return p


@dataclass
class Deps:
    mic: object
    player: object
    wake: object
    recorder: object
    stt: object
    tts: object
    log: object
    brain: object


def build_deps(cfg: VoiceConfig, room: bool) -> Deps:
    """Construct every real dependency from config. Loads all local models."""
    from jarvis.voice.audio import MicStream, Player
    from jarvis.voice.brain import Brain
    from jarvis.voice.room import build_room_server
    from jarvis.voice.speech import KokoroTTS
    from jarvis.voice.stt import WhisperSTT, load_vocab_prompt
    from jarvis.voice.turn import SileroVAD, SmartTurn, UtteranceRecorder
    from jarvis.voice.voicelog import VoiceLog
    from jarvis.voice.wakeword import WakeWord

    wake = WakeWord(cfg.wake_models_dir, cfg.wake_threshold)
    print(f"[voice] wake words: {', '.join(wake.names)}", file=sys.stderr, flush=True)
    recorder = UtteranceRecorder(SileroVAD(), SmartTurn(cfg.smart_turn_model),
                                 hard_silence_s=cfg.hard_silence_s,
                                 pause_check_s=cfg.pause_check_s,
                                 max_utterance_s=cfg.max_utterance_s)
    stt = WhisperSTT(cfg.stt_model, load_vocab_prompt(cfg.vocab_file))
    tts = KokoroTTS(cfg.kokoro_model, cfg.kokoro_voices, cfg.tts_voice)
    extra_mcp = None if not room else {"room": build_room_server(cfg.room_api_url)}
    return Deps(
        mic=MicStream(cfg.input_device),
        player=Player(cfg.output_device),
        wake=wake,
        recorder=recorder,
        stt=stt,
        tts=tts,
        log=VoiceLog(cfg.vault, cfg.data_dir),
        brain=Brain(cfg, extra_mcp=extra_mcp),
    )


def warm_stt(stt: object) -> None:
    """Load the Whisper weights now so the first turn is not slowed by it."""
    stt.transcribe(np.zeros(16000, dtype=np.int16))  # type: ignore[attr-defined]


async def run(cfg: VoiceConfig, room: bool) -> None:
    from jarvis.voice.loop import VoiceLoop

    d = build_deps(cfg, room)
    await asyncio.to_thread(warm_stt, d.stt)
    loop = VoiceLoop(d.mic, d.player, d.wake, d.recorder, d.stt, d.brain, d.tts,
                     d.log, cfg)
    d.brain.confirm = loop.confirm  # type: ignore[attr-defined]
    try:
        print("[voice] starting brain", file=sys.stderr, flush=True)
        await d.brain.start()  # type: ignore[attr-defined]
        print("[voice] ready", file=sys.stderr, flush=True)
        await loop.run()
    finally:
        d.player.close()  # type: ignore[attr-defined]
        await loop.aclose()
        await d.brain.close()  # type: ignore[attr-defined]


async def check(cfg: VoiceConfig, room: bool) -> None:
    d = build_deps(cfg, room)
    await asyncio.to_thread(warm_stt, d.stt)
    d.tts.synth("ok")  # type: ignore[attr-defined]
    await d.brain.close()  # type: ignore[attr-defined]


def list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        list_devices()
        return 0
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                        format="[voice] %(name)s %(levelname)s %(message)s")
    logging.getLogger("jarvis").setLevel(logging.INFO)
    # Kokoro's phonemizer warns on every sentence about word-count mismatches;
    # harmless. It resets its own level on each call, so filter instead.
    logging.getLogger("phonemizer").addFilter(lambda r: r.levelno >= logging.ERROR)
    cfg = resolve_paths(
        load_config(resolve_config_path(args.config), resolve_config_path(args.local)))
    room = not args.no_room
    try:
        if args.check:
            asyncio.run(check(cfg, room))
            print("ok")
        else:
            asyncio.run(run(cfg, room))
    except KeyboardInterrupt:
        print("[voice] stopped", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
