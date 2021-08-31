import click
from .cli import cli

from ..interface import console

from ..program import load
from ..inference import Optimizer, minibatch, DirectEmbedding

@cli.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-e", "--epochs", default=1, help="Number of training epochs")
@click.option("-l", "--learning-rate", default=0.01, show_default=True, help="Optimizer learning rate")
@click.option("-b", "--batch-size", default=1, help="Batch size")
@click.option("-i", "--instrument", type=click.Path(), help="Output file for instrumentation logs")
@click.option("-r", "--resolution", default=1, help="Instrumentation resolution (in epochs)")
def train(filename, epochs, learning_rate, batch_size, instrument, resolution):
    """Train FILENAME with the provided parameters."""

    # load the program and build the optimizer
    program, evidence = load(filename)

    optimizer = Optimizer(program, optimizer="adam", learning_rate=learning_rate)

    # instrumenter = Instrumenter(instrument, context={
    #     "seed" : seed(),
    #     "benchmark" : filename,
    #     "epochs" : epochs,
    #     "optimizer" : optimizer,
    #     "learning-rate" : learning_rate
    # })

    # train
    for batch in minibatch(evidence, batch_size, epochs=epochs):
        with optimizer as opt:
            log_prob = batch.log_prob(program, DirectEmbedding())
            opt.maximize(log_prob)

        if batch.index % resolution == 0:
            log = {
                "epoch" : batch.epoch
            }
            # instrumenter.emit(**log)

    if instrument:
        # instrumenter.flush()
        pass

    console.print("RESULTS")
    for parameter in program._parameters:
        console.print(f"{parameter.name}: {parameter.value:f}")