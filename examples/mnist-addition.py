"""Sherlog example: MNIST digit addition."""

import click

from random import choice
from sherlog.program import loads
from sherlog.inference import minibatch, FunctionalEmbedding, Optimizer
from sherlog.interface import print, initialize
from sherlog.interface.instrumentation import minotaur

from torch import nn, argmax

import torchvision
import torchvision.transforms as transforms

SOURCE = \
"""
digit(X; digit_nn[X]).

addition(X, Y; add[X', Y']) <- digit(X, X'), digit(Y, Y').
"""

class MNISTModule(nn.Module):
    def __init__(self):
        super(MNISTModule, self).__init__()

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

MNIST = torchvision.datasets.MNIST(
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
    x, xi = choice(MNIST)
    y, yi = choice(MNIST)

    return (x, y, xi + yi)

@click.command()
@click.option("--train", default=100, type=int, help="Number of training samples.")
@click.option("--test", default=100, type=int, help="Number of testing samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
@click.option("-i", "--instrumentation", type=str, help="Filepath to dump instrumentation logs.")
def cli(train, test, batch_size, epochs, learning_rate, instrumentation):
    
    # do this first, or we lose some initial messages
    if instrumentation:
        minotaur.add_filepath_handler(instrumentation)

    minotaur.enter("mnist addition")

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
        "digit_nn" : MNISTModule()
    })

    # load the data
    print(f"Generating {train} training points...")

    with minotaur("data generation"):
        data = [sample() for _ in range(train)]

        embedder = FunctionalEmbedding(
            evidence=lambda p: f"addition(left, right, {p[-1]})",
            parameters=lambda p: {
                "left" : p[0],
                "right" : p[1]
            }
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

            print(f"Batch {batch.index}:{batch.epoch} loss: {batch_loss.item()}.")


    # check the accuracy of the digit classifier
    correct = 0
    for _ in range(test):
        image, d = choice(MNIST)
        result = argmax(program._locals["digit_nn"](image)).item()

        if image == d:
            correct += 1

    print(correct / test)
    
    minotaur.exit()

if __name__ == "__main__":
    cli()