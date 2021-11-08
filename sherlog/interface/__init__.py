"""Contains infrastructure facilitating communication with the OCaml Sherlog server."""

from . import socket, server
from .console import print

from minotaur import Minotaur
from typing import Optional

# GLOBALS

_SOCKET = None
minotaur = Minotaur()

# INITIALIZATION

def initialize(port : int, instrumentation : Optional[str] = None) -> Minotaur:
    """Initialize Sherlog.
    
    This function must be called before executing functions in `sherlog.interface`.
    
    Parameters
    ----------
    port : int
        The Sherlog query server will serve requests on this port.

    instrumentation : str, optional
        If given, instrumentation will be saved to the given file.        
    """

    # if given instrumentation, configure minotaur appropriately
    global minotaur

    if instrumentation:
        minotaur.add_filepath_handler(instrumentation)

    # initialize the query server on the appropriate socket
    global _SOCKET

    server.initialize_server(port)
    
    while not _SOCKET:
        try:
            _SOCKET = socket.connect(port)
        except Exception as e:
            pass

# EXCEPTIONS

class CommunicationError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

# COMMUNICATION METHODS

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

def query(program, evidence, width = None):
    """Query a Sherlog program to explain the provided evidence."""

    message = {
        "type" : "query-request",
        "program" : program.to_json(),
        "evidence" : evidence.to_json()
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
        return response["explanation"]