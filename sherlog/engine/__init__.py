from .socket import connect
from ..config import PORT
from time import sleep
from . import server

sleep(1)

class CommunicationError(Exception): pass

SOCKET = connect(PORT)

def echo(obj):
    message = {
        "command" : "echo",
        "message" : obj
    }
    return SOCKET.communicate(message)

def parse(string):
    message = {
        "command" : "parse",
        "message" : string
    }
    return SOCKET.communicate(message)

def register(program):
    message = {
        "command" : "register",
        "message" : program
    }
    return SOCKET.communicate(message)

def query(q):
    message = {
        "command" : "query",
        "message" : q
    }
    response = SOCKET.communicate(message)
    if response == "failure": raise CommunicationError()
    return response["model"], response["observations"]