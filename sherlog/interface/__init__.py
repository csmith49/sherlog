"""Sherlog.Interface"""

from .socket import connect
from .server import initialize_server
from ..config import PORT
from time import sleep
from . import server
from rich.console import Console

_SOCKET = None

def initialize(port=PORT):
    global _SOCKET
    initialize_server(port=port)
    while not _SOCKET:
        try:
            _SOCKET = connect(port)
        except Exception as e:
            pass

console = Console(markup=False)

class CommunicationError(Exception): pass

def parse(source: str):
    """Run a string through Sherlog's parser.

    Parameters
    ----------
    source : string
        Raw source text to be parsed

    Returns
    -------
    JSON representation of the encoded program

    Raises
    ------
    CommunicationError
    """
    message = {
        "command" : "parse",
        "program" : source
    }
    response = _SOCKET.communicate(message)
    if response == "failure": raise CommunicationError()
    return response

def query(program, query, depth=None, width=None, seeds=None):
    """Run a Sherlog program on a query.

    Parameters
    ----------
    program : JSON representation
    query : JSON representation

    depth : Optional[int]
    width : Optional[int]
    seeds : Optional[int]

    Returns
    -------
    JSON representation of the entailed model

    Raises
    ------
    CommunicationError
    """
    message = {
        "command" : "query",
        "program" : program,
        "query"   : query,
    }

    # build the extra config tags
    if depth:
        message["depth"] = depth
    if width:
        message["width"] = width
    if seeds:
        message["seeds"] = seeds

    # send and rec
    response = _SOCKET.communicate(message)
    if response == "failure":
        raise CommunicationError()
    elif response == "timeout":
        raise TimeoutError()
    else:
        return response
