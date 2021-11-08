"""
"""

import click

from sherlog.program import loads
from sherlog.inference import minibatch, Optimizer, FunctionalEmbedding
from sherlog.interface import print, initialize, minotaur

from random import choices
from torch import tensor
from collections import Counter

SOURCE = \
"""
!parameter w : unit[3].

trit(X; {-1, 0, 1} <~ categorical[w]).

addition(X, Y; add[X, Y]).

observe(T) <- trit(left, L), trit(right, R), addition(L, R, T).
"""

def sample(weights):
    trits = [-1, 0, 1]
    left = choices(trits, weights)[0]
    right = choices(trits, weights)[0]
    return {
        "left" : left,
        "right" : right
    }

@click.command()
@click.option("--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
@click.option("-s", "--samples", default=100, type=int, help="Number of per-datum samples.")
@click.option("-i", "--instrumentation", type=str, help="Instrumentation log filepath.")
@click.option("-f", "--forcing", is_flag=True, help="Enable evaluation execution forcing.")
@click.option("-c", "--caching", is_flag=True, help="Enable explanation sampling caching.")
def cli(train, batch_size, epochs, learning_rate, samples, instrumentation, forcing, caching):
    """Learn the parameters for a simple ternary digit adder."""

    # initialize!
    print("Initializing...")
    initialize(port=8007, instrumentation=instrumentation)

    minotaur.enter("ternary-addition")

    minotaur["train"] = train
    minotaur["batch-size"] = batch_size
    minotaur["epochs"] = epochs
    minotaur["learning-rate"] = learning_rate
    minotaur["samples"] = samples

    # load the program
    print("Loading the program...")
    program, _ = loads(SOURCE)

    # load the data
    print(f"Generating {train} training points...")

    weights = [0.2, 0.5, 0.3]

    data = [sample(weights) for _ in range(train)]

    embedder = FunctionalEmbedding(
        evidence=lambda s: f"observe({s['left'] + s['right']})"
    )

    # build the optimizer
    print(f"Initializing the optimizer with a learning rate of {learning_rate}...")
    optimizer = Optimizer(
        program=program,
        samples=samples,
        learning_rate=learning_rate,
        force=forcing,
        cache=caching
    )

    # iterate over the data and optimize
    old_batch_loss = tensor(0.0)

    for batch in minibatch(data, batch_size, epochs=epochs):
        with minotaur("batch"):
            # we print out a frame for each batch for debugging purposes...
            print(f"\n③ Batch {batch.index:03d} in Epoch {batch.epoch:03d} ③")

            minotaur["batch"] = batch.index
            minotaur["epoch"] = batch.epoch

            # lets see what's going on in the ground truth
            print("Batch GT:")

            total = len(batch.data) * 2
            values = Counter([datum["left"] for datum in batch.data] + [datum["right"] for datum in batch.data])
            print(f"w=[{values[-1] / total:.3f}, {values[0] / total:.3f}, {values[1] / total:.3f}]")

            minotaur["w-gt"] = [values[-1] / total, values[0] / total, values[1] / total]

            # okay, now let's optimize
            optimizer.maximize(*embedder.embed_all(batch.data))
            batch_loss = optimizer.optimize()

            # what is the batch loss?
            print(f"Batch loss: {batch_loss:.3f} (Δ={old_batch_loss - batch_loss:.3f})")

            # what are the learned parameters doing right now?
            print("Parameter Summary:")

            w = program.parameter("w")
            weights = (w / w.sum()).tolist()
            grad = w.grad.tolist()
            print(f"w=[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}], ∇(left_weights)=[{grad[0]:.3f}, {grad[1]:.3f}, {grad[2]:.3f}]")
            minotaur["w"] = weights
            minotaur["w-grad"] = grad

        old_batch_loss = batch_loss

    minotaur.exit()

if __name__ == "__main__":
    cli()