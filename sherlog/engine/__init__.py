from .socket import JSONSocket
from ..config import PORT
from . import server

class CommunicationError(Exception): pass

def echo(obj):
    message = {
        "command" : "echo",
        "message" : obj
    }
    with JSONSocket(PORT) as s:
        s.send(message)
        response = s.receive()
    return response

def parse(string):
    message = {
        "command" : "parse",
        "message" : string
    }
    with JSONSocket(PORT) as s:
        s.send(message)
        response = s.receive()
    return response

def register(program):
    message = {
        "command" : "register",
        "message" : program
    }
    with JSONSocket(PORT) as s:
        s.send(message)

def query(q):
    message = {
        "command" : "query",
        "message" : q
    }
    with JSONSocket(PORT) as s:
        s.send(message)
        response = s.receive()
    if response == "failure":
        raise CommunicationError()
    return response["model"], response["observations"]