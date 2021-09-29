"""Sherlog example: sort.

TODO - instrumentation and performance.
"""

import click

from random import randint
from sherlog.program import loads
from sherlog.inference import minibatch, FunctionalEmbedding, Optimizer
from sherlog.interface import print, initialize
from sherlog.interface.instrumentation import minotaur

from typing import Tuple, List
from torch import nn, zeros, argmax

SOURCE = \
"""
swap_rate(X, Y; swap_nn[X, Y]).
swap(X, Y; categorical[R]) <- swap_rate(X, Y, R).

hole(X, Y, X, Y) <- swap(X, Y, 0).
hole(X, Y, Y, X) <- swap(X, Y, 1).

bubble(X :: [], [], X).
bubble(A :: B :: L, X :: R, Y) <- hole(A, B, X, Y), bubble(Y :: L, R, X).

bubblesort([], L, L).
bubblesort(L, R, S) <- bubble(L, L', X), bubblesort(L', X :: R, S).

sort(L, L') <- bubblesort(L, [], L').
"""

# sampling lists

def sample(length : int, min : int = 0, max : int = 9) -> Tuple[List[int], List[int]]:
    original = [randint(min, max) for _ in range(length)]
    result = sorted(original)

    return (original, result)

# swap predictor

class SwapModule(nn.Module):
    def __init__(self):
        super(SwapModule, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(20, 2),
            nn.Softmax(1)
        )

    def forward(self, x, y):
        input = zeros(20)
        input[int(x)] = 1.0
        input[int(y)] = 1.0
        output = self.nn(input.unsqueeze(0))
        return output.squeeze(0)

@click.command()
@click.option("--length", default=3, type=int, help="Length of lists in training set.")
@click.option("--train", default=100, type=int, help="Number of training samples.")
@click.option("--test", default=100, type=int, help="Number of testing samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
@click.option("-i", "--instrumentation", type=str, help="Filepath to save instrumentation logs.")
def cli(length, train, test, batch_size, epochs, learning_rate, instrumentation):
    """Train a program to sort lists."""

    # do this first, or we lose some initial messages
    if instrumentation:
        minotaur.add_filepath_handler(instrumentation)

    minotaur.enter("sort")

    minotaur["length"] = length
    minotaur["train"] = train
    minotaur["batch size"] = batch_size
    minotaur["epochs"] = epochs
    minotaur["learning rate"] = learning_rate

    # initialize!
    print("Initializing...")
    initialize(port=8007)

    # load the program
    
    print("Loading the program...")
    program, _ = loads(SOURCE, locals={
        "swap_nn" : SwapModule()
    })

    # load the data
    print(f"Generating {train} training points...")

    with minotaur("data generation"):
        data = [sample(length) for _ in range(train)]
        embedder = FunctionalEmbedding(
            evidence=lambda p: f"sort({p[0]}, {p[1]})"
        )

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

            print(f"Batch {batch.index}:{batch.epoch} loss: {batch_loss}.")

    # benchmark the accuracy of the learned module
    print("Benchmarking the learned function approximator...")

    correct = 0
    for _ in range(test):
        left, right = randint(0, 9), randint(0, 9)
        result = argmax(program._locals["swap_nn"](left, right)).item()
        target = 0 if left <= right else 1
        if result == target:
            correct += 1
        
    print(f"Accuracy: {correct / test}.")
    minotaur.exit()

if __name__ == "__main__":
    cli()