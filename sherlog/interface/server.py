"""On import, starts an instance of `sherlog-server` on port `config.PORT`.

Relies on `atexit` to terminate the server when this module goes out of scope."""

import atexit
import logging
from subprocess import Popen

# maximum timeout per sample attempt
TIMEOUT = 10

_SERVER = None

logger = logging.getLogger("sherlog.interface.server")

def close_server():
    """Send the termination signal to the server."""
    
    global _SERVER
    logger.info("Terminating the translation server...")
    if _SERVER:
        _SERVER.terminate()
    logger.info("Translation server terminated.")

def initialize_server(port : int):
    """Initialize the OCaml Sherlog server on the indicated port."""

    logger.info("Starting translation server on port %i...", port)
    global _SERVER
    _SERVER = Popen(["sherlog-server", "--port", f"{port}", "--timeout", f"{TIMEOUT}"])
    logger.info("Translation port successfully started on port %i.", port)

atexit.register(close_server)