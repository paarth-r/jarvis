"""
LLM agent subprocess.
Reads JSON-lines IntentEvents from stdin; writes JSON-lines ActionEvents to stdout.
This is a stub — Claude API integration is a future sub-project.
"""
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                    format="[llm] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("llm_agent started (stub)")
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
            logger.info("llm_agent shutting down")
            break
        else:
            logger.info("llm_agent stub — ignoring: %s", cmd)

    sys.exit(0)


if __name__ == "__main__":
    main()
