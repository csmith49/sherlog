"""Defines simple logging functionality.

Notes
-----
Use `get(module_name)` to acquire a logging object for module `module_name`.
"""

import logging
import logging.config
from rich.logging import RichHandler

FORMAT = "%(message)s"
HANDLER = RichHandler()

logging.basicConfig(
    level="WARNING",
    format=FORMAT,
    handlers=[HANDLER]
)

PREFIX = "sherlog"
LOGGERS = {}

def get(module_name):
    """Acquires a logger for a module.
    
    Parameters
    ----------
    module_name : str
        Module name where the logger is being acquired.
    
    Returns
    -------
    logger
        Logger object specialized for the provided module.
    """
    logger = logging.getLogger(f"{PREFIX}.{module_name}")
    LOGGERS[module_name] = logger
    return logger

def logged_modules():
    """A list of all module names used to acquire a logger.

    Returns
    -------
    list[str]
    """
    return list(LOGGERS.keys())

def enable(*args):
    """Enables verbose output for all loggers from the provided module names.

    Parameters
    ----------
    *args : list[str]

    Notes
    -----
    Operates by setting the relevant logger's level to INFO.
    """
    for module_name in args:
        try:
            LOGGERS[module_name].setLevel(logging.INFO)
        except KeyError:
            pass