# Sherlog.Interface

from .socket import connect
from ..config import PORT
from .io import console
from time import sleep
from . import server

with console.status("Spinning up server..."):
    sleep(1)

class CommunicationError(Exception): pass

SOCKET = connect(PORT)

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
    response = SOCKET.communicate(message)
    if response == "failure": raise CommunicationError()
    return response

def query(program, query, depth=None, width=None, seeds=None):
    """Run a Sherlog program on a query.

    Parameters
    ----------
    program : JSON representation
    query : JSON representation

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
    if depth: message["depth"] = depth
    if width: message["width"] = width
    if seeds: message["seeds"] = seeds

    # send and rec
    response = SOCKET.communicate(message)
    if response == "failure": raise CommunicationError()
    return response
