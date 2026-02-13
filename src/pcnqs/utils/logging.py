from __future__ import annotations

import json
import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for CLI scripts and experiments."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_event(logger: logging.Logger, event: str, **payload: object) -> None:
    """Emit one-line structured JSON logs for experiment traces."""

    body = {"event": event, **payload}
    logger.info(json.dumps(body, sort_keys=True))
