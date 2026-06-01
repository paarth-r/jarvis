from __future__ import annotations
import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from .runtime.config import load_config
from .runtime.supervisor import Supervisor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_supervisor(config_path: str) -> Supervisor:
    cfg = load_config(config_path)
    return Supervisor(cfg)


async def _run(config_path: str) -> None:
    sup = build_supervisor(config_path)
    loop = asyncio.get_event_loop()

    def _handle_signal():
        logger.info("Shutdown signal received")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    try:
        await sup.start()
    except asyncio.CancelledError:
        logger.info("Jarvis stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jarvis runtime")
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run(str(config_path)))


if __name__ == "__main__":
    main()
