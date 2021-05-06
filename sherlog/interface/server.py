# Sherlog.Interface.Server
"""
On import, starts an instance of `sherlog-server` on port `config.PORT`.

Relies on `atexit` to terminate the server when this module goes out of scope.
"""

import atexit
from subprocess import Popen
from ..logs import get
from ..config import PORT, TIMEOUT


logger = get("interface.server")

SERVER_ARGS = ["sherlog-server", "--port", f"{PORT}", "--timeout", f"{TIMEOUT}"]
logger.info("Starting translation server on port %i...", PORT)
SERVER = Popen(SERVER_ARGS)
logger.info("Translation port successfully started on port %i.", PORT)

def close_server():
    """Send the termination signal to the server."""
    logger.info("Terminating the translation server...")
    SERVER.terminate()
    logger.info("Translation server terminated.")

atexit.register(close_server)
