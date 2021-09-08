"""Sherlog example: truncated Gaussian.

TODO - instrumentation and performance.
"""

from sherlog.inference.embedding import FunctionalEmbedding
import click
from torch import tensor

from random import gauss
from sherlog.program import loads
from sherlog.inference import minibatch, Optimizer
from sherlog.interface import console, initialize

SOURCE = \
"""
# parameters
!parameter mean: real.

# rules
sample(ID; normal[mean, 1.0]).
clipped(ID, X; gt[X, threshold]) <- sample(ID, X).
"""

@click.command()
@click.option("--mean", type=float, default=0.0, help="Mean of the underlying Gaussian.")
@click.option("--threshold", type=float, default=0.5, help="Threshold for clipping random values.")
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
def cli(mean, threshold, train, batch_size, epochs, learning_rate):
    
    # initialize!
    initialize(port=8007)

    # load the program
    program, _ = loads(SOURCE)

    # load the data
    def sample(mean, threshold):
        x = gauss(mean, 1.0)
        if x >= threshold:
            return x
        else:
            return sample(mean, threshold)

    data = [sample(mean, threshold) for _ in range(train)]
    embedder = FunctionalEmbedding(
        evidence=lambda s: f"sample(id, {s})",
        conditional=lambda s: f"clipped(id, {s}, 1.0)",
        parameters=lambda _: {
            "threshold" : tensor(threshold)
        }
    )

    # build the optimizer
    optimizer = Optimizer(program, learning_rate=learning_rate)

    # iterate over the data, and optimize
    for batch in minibatch(data, batch_size, epochs=epochs):
        optimizer.maximize(*embedder.embed_all(batch.data))
        optimizer.optimize()

    # report the parameter
    for parameter in program._parameters:
        console.print(parameter.name, parameter.value)

if __name__ == "__main__":
    cli()