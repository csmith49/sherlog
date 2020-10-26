from ..logs import get
from subprocess import Popen
from ..config import PORT
import atexit

logger = get("engine.server")

SERVER_ARGS = ["./server", "--port", f"{PORT}"]
logger.info(f"Starting translation server on port {PORT}...")
SERVER = Popen(SERVER_ARGS)
logger.info(f"Translation port successfully started on port {PORT}.")

def close_server():
    logger.info(f"Terminating the translation server...")
    SERVER.terminate()
    logger.info(f"Translation server terminated.")

atexit.register(close_server)