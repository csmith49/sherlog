"""
"""

import click

from sherlog.program import loads
from sherlog.inference import minibatch, Optimizer, FunctionalEmbedding
from sherlog.interface import print, initialize

from random import choices, random
from enum import Enum
from torch import tensor, softmax

from collections import Counter

SOURCE = \
"""
# parameters
!parameter weight : unit.
!parameter urn_one_weights : unit[2].
!parameter urn_two_weights : unit[3].

flip(coin; {tails, heads} <~ bernoulli[weight]).
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

game(R) <-
    flip(coin, F),
    draw(urn_one, B1),
    draw(urn_two, B2),
    outcome(F, B1, B2, R).
"""

# coin
class Coin(Enum):
    TAILS = 0
    HEADS = 1

def sample_coin(weight):
    if random() <= weight:
        return Coin.HEADS
    else:
        return Coin.TAILS

# color
class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2

def sample_color(weights):
    return choices([Color.RED, Color.GREEN, Color.BLUE], weights)

# outcomes
def outcome(coin : Coin, ball1 : Color, ball2 : Color) -> bool:
    if (coin == Coin.HEADS) and (ball1 == Color.RED or ball2 == Color.RED):
        return True
    elif (ball1 == ball2):
        return True
    else:
        return False

# cli
@click.command()
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
def cli(train, batch_size, epochs, learning_rate):
    """Learn the parameters of a simple coin-ball problem."""

    # initialize!
    print("Initializing...")
    initialize(port=8007)

    # load the program
    print("Loading the program...")
    program, _ = loads(SOURCE, locals={})

    # generate the data and embedding
    print(f"Generating {train} training points...")

    def sample():
        coin = choices([Coin.TAILS, Coin.HEADS], [0.5, 0.5])[0]
        color1 = choices([Color.RED, Color.BLUE], [0.7, 0.3])[0]
        color2 = choices([Color.RED, Color.GREEN, Color.BLUE], [0.4, 0.3, 0.3])[0]

        return {
            "coin" : coin,
            "color1" : color1,
            "color2" : color2,
            "outcome" : outcome(coin, color1, color2)
        }

    data = [sample() for _ in range(train)]
    embedder = FunctionalEmbedding(
        evidence=lambda s: "game(win)" if s["outcome"] else "game(loss)"
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
        print(f"\n🪙 Batch {batch.index:03d} in Epoch {batch.epoch:03d} 🪙")

        # what is the parameterization implied by the batch?
        print("Batch GT:")

        coins = Counter([datum["coin"] for datum in batch.data])
        print(f"weight={coins[Coin.HEADS] / len(batch.data):.3f}")

        color1s = Counter([datum["color1"] for datum in batch.data])
        print(f"urn_one_weights=[{color1s[Color.RED] / len(batch.data):.3f}, {color1s[Color.BLUE] / len(batch.data):.3f}]")
        
        color2s = Counter([datum["color2"] for datum in batch.data])
        print(f"urn_two_weights=[{color2s[Color.RED] / len(batch.data):.3f}, {color2s[Color.BLUE] / len(batch.data):.3f}, {color2s[Color.GREEN] / len(batch.data):.3f}]")

        # okay, now let's optimize
        optimizer.maximize(*embedder.embed_all(batch.data))
        batch_loss = optimizer.optimize()

        # what is the batch loss?
        print(f"Batch loss: {batch_loss:.3f} (Δ={old_batch_loss - batch_loss:.3f})")

        # and what are the program parameters doing?
        print("Parameter summary:")

        weight = program.parameter("weight")
        print(f"weight={weight.item():.3f}, ∇(weight)={weight.grad.item():.3f}")

        urn_one_weights = program.parameter("urn_one_weights")
        weights, grad = softmax(urn_one_weights, dim=0).tolist(), urn_one_weights.grad.tolist()
        print(f"urn_one_weights=[{weights[0]:.3f}, {weights[1]:.3f}], ∇(urn_one_weights)=[{grad[0]:.3f}, {grad[1]:.3f}]")

        urn_two_weights = program.parameter("urn_two_weights")
        weights, grad = softmax(urn_two_weights, dim=0).tolist(), urn_two_weights.grad.tolist()
        print(f"urn_two_weights=[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}], ∇(urn_two_weights)=[{grad[0]:.3f}, {grad[1]:.3f}, {grad[2]:.3f}]")

        old_batch_loss = batch_loss

if __name__ == "__main__":
    cli()