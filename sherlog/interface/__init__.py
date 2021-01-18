from .socket import connect
from ..config import PORT
from time import sleep
from . import server

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

def query(program, query, max_depth=None):
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
        "command" : "run",
        "program" : program,
        "query"   : query,
        "depth"   : max_depth
    }
    response = SOCKET.communicate(message)
    if response == "failure": raise CommunicationError()
    return response
