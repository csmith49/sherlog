import click
from rich.table import Table
from .cli import cli

from ..interface import print
from ..interface.instrumentation import minotaur

from ..program import load
from ..inference import Optimizer, minibatch, DirectEmbedding

@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-e", "--epochs", default=1, help="Number of training epochs")
@click.option("-l", "--learning-rate", default=0.01, show_default=True, help="Optimizer learning rate")
@click.option("-b", "--batch-size", default=1, help="Batch size")
@minotaur("train")
def train(filename, epochs, learning_rate, batch_size):
    """Train FILENAME with the provided parameters."""

    minotaur["epochs"] = epochs
    minotaur["learning rate"] = learning_rate
    minotaur["batch size"] = batch_size

    # load the program and build the optimizer
    program, evidence = load(filename)

    optimizer = Optimizer(program, learning_rate=learning_rate)
    embedder = DirectEmbedding()

    # train
    for batch in minibatch(evidence, batch_size, epochs=epochs):
        with minotaur("batch"):
            optimizer.maximize(*embedder.embed_all(batch.data))
            optimizer.optimize()

            minotaur["batch"] = batch.index
            minotaur["epoch"] = batch.epoch

    # output program parameters in fancy table
    parameter_table = Table(title="Inferred Parameters")
    parameter_table.add_column("Name")
    parameter_table.add_column("Value")
    parameter_table.add_column("Domain")

    for parameter in program._parameters:
        parameter_table.add_row(parameter.name, str(parameter.value), parameter.domain)

    print(parameter_table)