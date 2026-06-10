"""Shared logging configuration for Kuchikae."""

from __future__ import annotations

import logging
import sys


def setup_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger
