from socket import socket, AF_INET, SOCK_STREAM
from json import loads, dumps
from ..logs import get

logger = get("engine.socket")

# exception for the socket layer
class SocketNotBound(Exception):
    pass

# simple wrapper around sockets, built for sending and receiving JSON objects
class JSONSocket:
    def __init__(self, port):
        self.port = port
        self._socket = socket(AF_INET, SOCK_STREAM)
        self._is_connected = False
        logger.info(f"Socket initialized on port {self.port}.")

    # context manager portion just handles closing connection when needed
    def __enter__(self):
        self._socket.connect( ('localhost', self.port) )
        self._is_connected = True
        logger.info(f"Socket connected to server on port {self.port}.")
        return self

    def __exit__(self, *args):
        self._is_connected = False
        self._socket.close()
        logger.info(f"Socket disconnected from server on port {self.port}.")

    # wrapping around send/recv to avoid having to worry about message lengths
    def send(self, obj):
        # can't send if there's nothing bound
        if not self._is_connected:
            raise SocketNotBound()
        # convert the obj to a json string and send it on the way
        message = dumps(obj) + "\n"
        self._socket.send(message.encode())
        logger.info(f"Message sent to server on port {self.port}...")

    def receive(self, buffer_size=1024):
        # can't receive from nothing
        if not self._is_connected:
            raise SocketNotBound()
        # we'll poll repeatedly until we get a complete json object
        message = ""
        while True:
            message += self._socket.recv(buffer_size).decode()
            try:
                obj = loads(message)
                logger.info(f"Response retrieved from server on port {self.port}.")
                return obj
            except: pass