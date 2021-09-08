"""Sherlog example: coin flip.

TODO - instrumentation and performance.
"""

import click

from random import random
from sherlog.program import loads
from sherlog.inference import minibatch, PartitionEmbedding, Optimizer
from sherlog.interface import console, initialize

SOURCE = \
"""
# parameters
!parameter p: unit.

# rules
flip(coin; bernoulli[p]).
result(heads) <- flip(coin, 1.0).
result(tails) <- flip(coin, 0.0).
"""

@click.command()
@click.option("-p", "--probability", type=float, default=0.6, help="Probability of a coin flip coming up heads.")
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
def cli(probability, train, batch_size, epochs, learning_rate):
    
    # initialize!
    initialize(port=8007)

    # load the program
    program, _ = loads(SOURCE)

    # load the data
    data = [random() <= probability for _ in range(train)]
    embedder = PartitionEmbedding({
        True: "result(heads)",
        False: "result(tails)"
    })

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