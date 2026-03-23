"""Central logging setup for CLI and Telegram entrypoints."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once (idempotent if handlers already exist)."""
    if isinstance(level, str):
        numeric = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric = level
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(numeric)
