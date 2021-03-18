# Sherlog.Interface.Server
"""
On import, starts an instance of `sherlog-server` on port `config.PORT`.

Relies on `atexit` to terminate the server when this module goes out of scope.
"""

from ..logs import get
from subprocess import Popen
from ..config import PORT
import atexit

logger = get("interface.server")

SERVER_ARGS = ["sherlog-server", "--port", f"{PORT}"]
logger.info(f"Starting translation server on port {PORT}...")
SERVER = Popen(SERVER_ARGS)
logger.info(f"Translation port successfully started on port {PORT}.")

def close_server():
    logger.info(f"Terminating the translation server...")
    SERVER.terminate()
    logger.info(f"Translation server terminated.")

atexit.register(close_server)