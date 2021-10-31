"""
"""

import click

from sherlog.program import loads
from sherlog.inference import minibatch, Optimizer, FunctionalEmbedding
from sherlog.interface import print, initialize

from random import choice
from torch import nn, tensor, argmax

import torchvision
import torchvision.transforms as transforms

SOURCE = \
"""
digit(X; {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} <- digit_nn[X]).
observe(I, D) <- digit(I, D).
"""

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()

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
    image, digit = choice(MNIST)
    return {
        "image" : image,
        "digit" : digit
    }

def classification_accuracy(classifier, data):
    correct = 0
    for datum in data:
        prediction = argmax(classifier(datum["image"]))
        if prediction == datum["digit"]:
            correct += 1
    return correct / len(data)

@click.command()
@click.option("--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
def cli(train, batch_size, epochs, learning_rate):
    """Learn the parameters for a simple MNIST digit classifier."""
    
    # initialize!
    print("Initializing...")
    initialize(port=8007)

    # load the program
    print("Loading the program...")
    program, _ = loads(SOURCE, locals={"digit_nn" : MNISTClassifier()})

    # load the data
    print(f"Generating {train} training points...")

    data = [sample() for _ in range(train)]

    embedder = FunctionalEmbedding(
        evidence=lambda s: f"observe(image, {s['digit']})",
        parameters=lambda s: {
            "image" : s["image"]
        }
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
        # we print out a frame for each batch for debugging purposes...
        print(f"\n✏️ Batch {batch.index:03d} in Epoch {batch.epoch:03d} ✏️")

        # okay, now let's optimize
        optimizer.maximize(*embedder.embed_all(batch.data))
        batch_loss = optimizer.optimize()

        # what is the batch loss?
        print(f"Batch loss: {batch_loss:.3f} (Δ={old_batch_loss - batch_loss:.3f})")

        # what is the network doing wrt accuracy?
        classifier = program._locals["digit_nn"]
        accuracy = classification_accuracy(classifier, batch.data)
        print(f"Batch accuracy: {accuracy:0.3f}")

        old_batch_loss = batch_loss

if __name__ == "__main__":
    cli()