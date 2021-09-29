"""Sherlog example: coin-ball problem.

TODO - instrumentation and performance.
"""

import click

from sherlog.program import loads
from sherlog.inference import minibatch, Optimizer, FunctionalEmbedding
from sherlog.interface import initialize
from sherlog.interface.instrumentation import instrument, minotaur

from torch import nn
from torch.nn.functional import softmax
from random import choices, gauss
from enum import Enum, auto
from colorsys import rgb_to_hsv, hsv_to_rgb

from rich import print

SOURCE = \
"""
face(C; {tails, heads} <- coin_nn[C]).
color(RGB; {red, green, blue} <- color_nn[RGB]).

# parameters
!parameter urn_one_weights : unit[2].
!parameter urn_two_weights : unit[3].

draw(urn_one; {red, blue} <~ categorical[urn_one_weights]).
draw(urn_two; {red, green, blue} <~ categorical[urn_two_weights]).

outcome(heads, red, red, win).
outcome(heads, red, blue, win).
outcome(heads, red, green, win).
outcome(heads, blue, red, win).
outcome(heads, blue, blue, win).
outcome(heads, blue, green, loss).
outcome(tails, red, red, win).
outcome(tails, red, blue, loss).
outcome(tails, red, green, loss).
outcome(tails, blue, red, loss).
outcome(tails, blue, blue, win).
outocme(tails, blue, green, loss).

game(C, U1, U2, R) <-
    face(C, F),
    draw(urn_one, C1), color(U1, C1),
    draw(urn_two, C2), color(U2, C2),
    outcome(F, C1, C2, R).
"""

# coin
class Coin(Enum):
    TAILS = auto(), (1.0, 0.0)
    HEADS = auto(), (0.0, 1.0)

    def noise(self, scaling : float = 0.03):
        t, h = self.value[-1]
        t = max(0.0, min(1.0, gauss(t, scaling)))
        h = max(0.0, min(1.0, gauss(h, scaling)))
        return (t, h)

# color
class Color(Enum):
    RED = auto(), (1.0, 0.0, 0.0)
    GREEN = auto(), (0.0, 1.0, 0.0)
    BLUE = auto(), (0.0, 0.0, 1.0)

    def noise(self, scaling : float = 0.03):
        """Add noise to the RGB representation of the color."""

        h, s, v = rgb_to_hsv(*self.value[-1])
        h = gauss(h, scaling) % 1.0
        s = max(0.0, min(1.0, gauss(s, scaling)))
        v = max(0.0, min(1.0, gauss(v, scaling)))
        return hsv_to_rgb(h, s, v)

# outcomes
def outcome(coin : Coin, ball1 : Color, ball2 : Color) -> bool:
    if (coin == Coin.HEADS) and (ball1 == Color.RED or ball2 == Color.RED):
        return True
    elif (ball1 == ball2):
        return True
    else:
        return False

class CoinModule(nn.Module):
    def __init__(self):
        super(CoinModule, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.classifier(x.unsqueeze(0)).float().squeeze(0)


class ColorModule(nn.Module):
    def __init__(self):
        super(ColorModule, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.classifier(x.unsqueeze(0)).float().squeeze(0)

# cli

@click.command()
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
@click.option("-i", "--instrumentation", type=str, help="Instrumentation log destination.")
def cli(train, batch_size, epochs, learning_rate, instrumentation):
    
    # make sure we do this first...
    if instrumentation:
        minotaur.add_filepath_handler(instrumentation)

    # initialize!
    initialize(port=8007)

    minotaur["train"] = train
    minotaur["batch size"] = batch_size
    minotaur["epochs"] = epochs
    minotaur["learning rate"] = learning_rate

    # load the program
    with minotaur("program load"):
        program, _ = loads(SOURCE, locals={
            "coin_nn" : CoinModule(),
            "color_nn" : ColorModule()
        })

    # load the data
    def sample():
        coin = choices([Coin.TAILS, Coin.HEADS], [0.5, 0.5])[0]
        ball1 = choices([Color.RED, Color.BLUE], [0.7, 0.3])[0]
        ball2 = choices([Color.RED, Color.GREEN, Color.BLUE], [0.4, 0.3, 0.3])[0]
        
        return (coin.noise(), ball1.noise(), ball2.noise(), outcome(coin, ball1, ball2))

    with minotaur("data load"):
        data = [sample() for _ in range(train)]
        embedder = FunctionalEmbedding(
            evidence=lambda p: "game(coin, ball1, ball2, win)" if p[-1] else "game(coin, ball1, ball2, loss)",
            parameters=lambda p: {
                "coin" : p[0],
                "ball1" : p[1],
                "ball2" : p[2]
            }
        )

    # build the optimizer
    optimizer = Optimizer(program,
        learning_rate=learning_rate,
        explanations=10    
    )

    # iterate over the data and optimize
    for batch in minibatch(data, batch_size, epochs=epochs):
        with minotaur("batch"):
            optimizer.maximize(*embedder.embed_all(batch.data))
            batch_loss = optimizer.optimize()

            print(f"Batch {batch.index}:{batch.epoch} loss: {batch_loss.item()}")

            minotaur["batch"] = batch.index
            minotaur["epoch"] = batch.epoch

    # report the parameters
    for parameter in program._parameters:
        name, value = parameter.name, softmax(parameter.value, dim=0).tolist()
        minotaur[name] = value
        print(f"{name} : {value}")

if __name__ == "__main__":
    cli()