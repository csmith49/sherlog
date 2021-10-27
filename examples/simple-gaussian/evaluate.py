"""
"""

import click

from random import gauss
from sherlog.program import loads
from sherlog.inference import minibatch, FunctionalEmbedding, Optimizer
from sherlog.interface import print, initialize
from torch import tensor
import statistics as stats

SOURCE = \
"""
!parameter mu : real.
!parameter sigma : positive.

sample(S; normal[mu, sigma]).
observe(V) <- sample(_, V).
"""

@click.command()
@click.option("-m", "--mean", type=float, default=0.9, help="Mean of Gaussian.")
@click.option("-s", "--sdev", type=float, default=0.03, help="Standard deviation of Gaussian.")
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, help="Learning rate.")
def cli(mean, sdev, train, batch_size, epochs, learning_rate):
    """Train a simple Gaussian."""

    # initialize query server and instrumentation
    print("Initializing...")
    initialize(port=8007)

    # load the program
    print("Loading the program...")
    program, _ = loads(SOURCE)

    # generate the data and embedding
    print(f"Generating {train} training points...")
    data = [gauss(mean, sdev) for _ in range(train)]
    embedder = FunctionalEmbedding(
        evidence=lambda s: f"observe({s})"
    )

    # build the optimizer
    print(f"Initializing the optimizer with a learning rate of {learning_rate}...")
    optimizer = Optimizer(
        program=program,
        learning_rate=learning_rate
    )

    # iterate over the data and optimize
    old_batch_loss = tensor(0.0)

    for batch in minibatch(data, batch_size, epochs=epochs):
        # we print out a "frame" for each batch for debugging purposes...
        print(f"\nBatch {batch.index:03d} in Epoch {batch.epoch:03d}")

        # what is the distribution implied by the batch?
        print(f"Batch GT: μ={stats.mean(batch.data):.3f}, σ={stats.stdev(batch.data):.3f}")

        # okay, now let's optimize
        optimizer.maximize(*embedder.embed_all(batch.data))
        batch_loss = optimizer.optimize()

        # what is the batch loss?
        print(f"Batch loss: {batch_loss:.3f} (Δ={old_batch_loss - batch_loss:.3f})")

        # and what are the program parameters doing?
        print("Parameter summary:")

        mu = program.parameter("mu")
        print(f"μ={mu.item():.3f}, ∇μ={mu.grad.item():.3f}, error=±{abs(mu.item() - mean):.3f}")

        sigma = program.parameter("sigma")
        print(f"σ={sigma.item():.3f}, ∇σ={sigma.grad.item():.3f}, error=±{abs(sigma.item() - sdev):.3f}")

        old_batch_loss = batch_loss

if __name__ == "__main__":
    cli()