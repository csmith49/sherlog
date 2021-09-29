"""Sherlog example: coin flip.

TODO - instrumentation and performance.
"""

import click

from random import random
from sherlog.program import loads
from sherlog.inference import minibatch, PartitionEmbedding, Optimizer
from sherlog.interface import print, initialize
from sherlog.interface.instrumentation import minotaur

SOURCE = \
"""
!parameter p : unit.
flip(coin; {tails, heads} <~ bernoulli[p]).
result(F) <- flip(coin, F).
"""

@click.command()
@click.option("-p", "--probability", type=float, default=0.6, help="Probability of a coin flip coming up heads.")
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
@click.option("-i", "--instrumentation", type=str, help="Filepath to save instrumentation logs to.")
def cli(probability, train, batch_size, epochs, learning_rate, instrumentation):
    """Train a simple coin flip program."""

    # do this first, or we lose some initial messages
    if instrumentation:
        minotaur.add_filepath_handler(instrumentation)

    minotaur.enter("coin flip")

    minotaur["probability"] = probability
    minotaur["train"] = train
    minotaur["batch size"] = batch_size
    minotaur["epochs"] = epochs
    minotaur["learning rate"] = learning_rate

    # initialize!
    print("Initializing...")
    
    initialize(port=8007)

    # load the program
    print("Loading the program...")

    program, _ = loads(SOURCE)

    # load the data
    print(f"Generating {train} training points...")

    data = [random() <= probability for _ in range(train)]
    embedder = PartitionEmbedding({
        True: "result(heads)",
        False: "result(tails)"
    })

    # build the optimizer
    print(f"Initializing the optimizer with a learning rate of {learning_rate}...")

    optimizer = Optimizer(program, learning_rate=learning_rate)

    # iterate over the data, and optimize
    for batch in minibatch(data, batch_size, epochs=epochs):
        with minotaur("batch"):
            optimizer.maximize(*embedder.embed_all(batch.data))
            batch_loss = optimizer.optimize()

            minotaur["batch"] = batch.index
            minotaur["epoch"] = batch.epoch

            print(f"Batch {batch.index}:{batch.epoch} loss: {batch_loss.item()}.")

    # report the parameter
    p = program._parameters[0]
    minotaur["p"] = p.value.item()

    print(f"Resulting parameter p: {p.value}")

    minotaur.exit()

if __name__ == "__main__":
    cli()