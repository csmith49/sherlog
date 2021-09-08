"""Sherlog example: sort.

TODO - instrumentation and performance.
"""

from sherlog.inference.embedding import FunctionalEmbedding
import click
from torch import nn, zeros

from random import random, randint
from sherlog.program import loads
from sherlog.inference import minibatch, Optimizer
from sherlog.interface import console, initialize

from typing import Tuple, List

SOURCE = \
"""
swap_rate(X, Y; swap_nn[X, Y]).
swap(X, Y; categorical[R]) <- swap_rate(X, Y, R).

sort([], []).
sort(X :: XS, YS) <-
    partition(XS, X, Left, Right),
    sort(Left, LS),
    sort(Right, RS),
    append(LS, X :: RS, YS).

partition([], Y, [], []).
partition(X :: XS, Y, X :: LS, RS) <- swap(X, Y, 1), partition(XS, Y, LS, RS).
partition(X :: XS, Y, LS, X :: RS) <- swap(X, Y, 0), partition(XS, Y, LS, RS).

append([], YS, YS).
append(X :: XS, YS, X :: ZS) <- append(XS, YS, ZS).
"""

@click.command()
@click.option("--length", default=3, type=int, help="Length of lists in training set.")
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
def cli(length, train, batch_size, epochs, learning_rate):
    
    # initialize!
    initialize(port=8007)

    # load the program
    class Module(nn.Module):
        def __init__(self):
            super(Module, self).__init__()
            self.nn = nn.Sequential(
                nn.Linear(20, 2),
                nn.Softmax(1)
            )

        def forward(self, x, y):
            input = zeros(20)
            input[int(x)] = 1.0
            input[int(y) + 10] = 1.0
            output = self.nn(input.unsqueeze(0))
            return output.squeeze(0)

    program, _ = loads(SOURCE, locals={"swap_nn" : Module()})

    # load the data
    def sample(length : int, min : int = 0, max : int = 9) -> Tuple[List[int], List[int]]:
        original = [randint(min, max) for _ in range(length)]
        result = sorted(original)

        return (original, result)        

    data = [sample(length) for _ in range(train)]
    embedder = FunctionalEmbedding(
        evidence=lambda p: f"sort({p[0]}, {p[1]})"
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