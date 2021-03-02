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
    return logging.getLogger(f"sherlog.{module_name}")

def enable_verbose_output():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        if logger.name.startswith("sherlog."):
            logger.setLevel(logging.INFO)
