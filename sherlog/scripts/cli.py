import click

from .. import logs
from ..interface import console, initialize

@click.group()
@click.option("-l", "--log", multiple=True, type=click.Choice(logs.logged_modules(), case_sensitive=False), help="Enable verbose output via logging")
@click.option("-p", "--port", type=int)
def cli(log, port):
    """Wrapper for common manipulation of Sherlog programs."""

    # enable the provided logs
    if log:
        logs.enable(*log)
    
    # initialize the server at the indicated port
    if port is not None:
        initialize(port=port)
    else:
        initialize()
