import click
from rich.table import Table
from .cli import cli

from ..interface import console
from ..interface.instrumentation import Instrumenter, Timer, Seed

from ..program import load
from ..inference import Optimizer, minibatch, DirectEmbedding

@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-e", "--epochs", default=1, help="Number of training epochs")
@click.option("-l", "--learning-rate", default=0.01, show_default=True, help="Optimizer learning rate")
@click.option("-b", "--batch-size", default=1, help="Batch size")
@click.option("-i", "--instrument", type=click.Path(), help="Output file for instrumentation logs")
def train(filename, epochs, learning_rate, batch_size, instrument):
    """Train FILENAME with the provided parameters."""

    instrumenter = Instrumenter(
        filepath=instrument,
        context={
            "seed" : Seed(),
            "epochs" : epochs,
            "learning-rate" : learning_rate,
            "batch-size" : batch_size
        }
    )

    # load the program and build the optimizer
    program, evidence = load(filename)

    optimizer = Optimizer(program, learning_rate=learning_rate)
    embedder = DirectEmbedding()

    # train
    for batch in minibatch(evidence, batch_size, epochs=epochs):
        optimizer.maximize(*embedder.embed_all(batch.data))
        batch_loss = optimizer.optimize()

        if batch.index == 0:
            instrumenter.write(**{
                "epoch" : batch.epoch,
                "batch-objective" : (-1 * batch_loss).exp()
            })

    # output program parameters in fancy table
    parameter_table = Table(title="Inferred Parameters")
    parameter_table.add_column("Name")
    parameter_table.add_column("Value")
    parameter_table.add_column("Domain")

    for parameter in program._parameters:
        parameter_table.add_row(parameter.name, str(parameter.value), parameter.domain)

    console.print(parameter_table)