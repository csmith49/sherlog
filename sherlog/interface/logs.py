"""Defines simple logging functionality.

Notes
-----
Use `get(module_name)` to acquire a logging object for module `module_name`.
"""

import logging
import logging.config
from rich.logging import RichHandler
from typing import Iterable

FORMAT = "%(message)s"
HANDLER = RichHandler()

logging.basicConfig(
    level="WARNING",
    format=FORMAT,
    handlers=[HANDLER]
)

_PREFIX = "sherlog"
_VERBOSE_LOGGERS = {}

def get(name, verbose : bool = False):
    """Acquire a logger."""

    logger = logging.getLogger(f"{_PREFIX}.{name}")
    if verbose:
        _VERBOSE_LOGGERS[name] = logger
    return logger

def verbose_loggers() -> Iterable[str]:
    """A list of logger names supporting verbose output."""

    yield from _VERBOSE_LOGGERS.keys()

def enable(*loggers : str):
    """Enable verbose output for all identified loggers."""

    for logger in loggers:
        try:
            _VERBOSE_LOGGERS[logger].setLevel(logging.INFO)
        except KeyError:
            pass
