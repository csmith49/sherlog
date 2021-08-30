import click
from .cli import cli, console

from ..program import load

@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-b", "--burn-in", type=int, default=100)
def explain(filename, burn_in):
    """Explain FILENAME by sampling explanations per-evidence."""

    program, evidence = load(filename)

    for datum in evidence:
        console.print(f"Explaining {datum}:")
        explanation = program.sample_posterior(datum, burn_in=burn_in)
        console.print(f"Pipeline: {explanation.pipeline}")