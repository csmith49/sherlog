from rich.console import Console
import rich.progress as progress

console = Console()

def status(message):
    return console.status(message)

def track(iter, description=""):
    return progress.track(iter, description=description)

def print(message):
    return console.print(message)

def debug(message):
    return console.log(message)