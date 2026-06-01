"""
6-DOF arm controller subprocess.
Reads JSON-lines commands from stdin; writes JSON-lines status to stdout.
This is a stub — arm hardware integration is a future sub-project.
"""
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[arm] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("arm_controller started (stub)")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("bad JSON: %r", line)
            continue

        cmd = msg.get("cmd")
        if cmd == "shutdown":
            logger.info("arm_controller shutting down")
            break
        else:
            logger.info("arm_controller stub — ignoring cmd: %s", cmd)

    sys.exit(0)


if __name__ == "__main__":
    main()
