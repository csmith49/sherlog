import click

from ..interface import initialize
from ..interface.instrumentation import instrument as enable_instrumentation

@click.group()
@click.option("-i", "--instrument", type=str, help="Enable instrumentation via Minotaur")
@click.option("-p", "--port", type=int, default=8007)
def cli(instrument, port):
    """Wrapper for common manipulation of Sherlog programs."""

    # enable the provided logs
    if instrument:
        enable_instrumentation(instrument)

    # initialize the server at the indicated port
    initialize(port)
