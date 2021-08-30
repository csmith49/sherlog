"""Contains infrastructure facilitating communication with the OCaml Sherlog server."""

from .socket import connect
from .server import initialize_server
from ..config import PORT
from time import sleep
from . import server
from rich.console import Console

_SOCKET = None

def initialize(port=PORT):
    """Initialize the Sherlog server. A necessary prerequisite before executing functions in `sherlog.interface`.

    Parameters
    ----------
    port : int (default=PORT in `config.py`)
    """
    global _SOCKET
    initialize_server(port=port)
    while not _SOCKET:
        try:
            _SOCKET = connect(port)
        except Exception as e:
            pass

console = Console(markup=False)

class CommunicationError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

def parse_source(source : str):
    """Parse a Sherlog source file."""

    message = {
        "type" : "parse-source-request",
        "source" : source
    }

    response = _SOCKET.communicate(message)
    
    if response["type"] == "failure":
        raise CommunicationError(response["message"])
    elif response["type"] != "parse-source-response":
        raise CommunicationError(f"Found invalid response type {response['type']}.")
    else:
        return response["program"], response["evidence"]

def query(rules, evidence, posterior, width = None):
    """Query a Sherlog program to explain the provided evidence."""

    message = {
        "type" : "query-request",
        "program" : {
            "type" : "program",
            "rules" : rules,
            "parameters" : []
        },
        "evidence" : evidence,
        "posterior" : posterior
    }

    # add additional config stuff
    if width:
        message["search-width"] = width

    # send and rec
    response = _SOCKET.communicate(message)
    
    if response["type"] == "failure":
        raise CommunicationError(response["message"])
    elif response["type"] != "query-response":
        raise CommunicationError(f"Found invalid response type {response['type']}.")
    else:
        return response["explanations"]