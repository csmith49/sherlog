"""Sherlog example: MNIST digit addition.

TODO - unify with Sam's Pyro implementation.
"""

from sherlog.inference.embedding import FunctionalEmbedding, PartitionEmbedding
import click
from torch import nn

import torchvision
import torchvision.transforms as transforms

from random import choice
from sherlog.program import loads
from sherlog.inference import minibatch, Optimizer
from sherlog.interface import console, initialize

from typing import Tuple

SOURCE = \
"""
digit_probs(X; digit_nn[X]).
digit(X; categorical[P]) <- digit_probs(X, P).

addition(X, Y; add[X', Y']) <- digit(X, X'), digit(Y, Y').
"""

@click.command()
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
def cli(train, batch_size, epochs, learning_rate):
    
    # initialize!
    initialize(port=8007)

    # load the program
    class Module(nn.Module):
        def __init__(self):
            super(Module, self).__init__()

            self.encoder = nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.MaxPool2d(2, 2),
                nn.ReLU(True),
                nn.Conv2d(6, 16, 5),
                nn.MaxPool2d(2, 2),
                nn.ReLU(True)
            )
            self.classifier = nn.Sequential(
                nn.Linear(16 * 4 * 4, 120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 10),
                nn.Softmax(1)
            )
        
        def forward(self, x):
            z = self.encoder(x.unsqueeze(0)).view(-1, 16 * 4 * 4)
            return self.classifier(z).float().squeeze(0)

    program, _ = loads(SOURCE, locals={"digit_nn" : Module()})

    # load the data
    dataset = torchvision.datasets.MNIST(
        root="/tmp/MNIST",
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5,),
                (0.5,)
            )
        ])
    )

    def sample():
        x, xi = choice(dataset)
        y, yi = choice(dataset)

        return (x, y, xi + yi)

    data = [sample() for _ in range(train)]
    embedder = FunctionalEmbedding(
        evidence=lambda p: f"addition(left, right, {p[-1]})",
        parameters=lambda p: {
            "left" : p[0],
            "right" : p[1]
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