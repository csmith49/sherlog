import click

from ..interface import initialize, logs

@click.group()
@click.option("-v", "--verbose", multiple=True, type=str, help="Enable verbose output via logging")
@click.option("-p", "--port", type=int, default=8007)
def cli(verbose, port):
    """Wrapper for common manipulation of Sherlog programs."""

    # enable the provided logs
    if verbose:
        logs.enable(*verbose)
    
    # initialize the server at the indicated port
    initialize(port)
