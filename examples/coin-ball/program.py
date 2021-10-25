from sherlog.program import loads
from sherlog.inference import Optimizer
from torch import tensor, argmax, dist, softmax

from .modules import CoinModule, ColorModule


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

class Program:
    def __init__(self):
        self.program, _ = loads(SOURCE, locals={
            "coin_nn" : CoinModule(),
            "color_nn" : ColorModule()
        })

    def optimizer(self, learning_rate):
        return Optimizer(self.program, learning_rate=learning_rate)

    def urn_one_convergence(self, ground_truth):
        return dist(softmax(self.program._parameters[1].value, 0), softmax(tensor(ground_truth), 0))

    def urn_two_convergence(self, ground_truth):
        return dist(softmax(self.program._parameters[0].value, 0), softmax(tensor(ground_truth), 0))

    def coin_nn_performance(self, coins):
        correct = 0
        for coin in coins:
            prediction = self.program._locals["coin_nn"](tensor(coin.noise()))
            result = argmax(prediction)

            if coin.value[0] == result.item(): correct += 1

        return correct / len(coins)

    def color_nn_performance(self, colors):
        correct = 0
        for color in colors:
            prediction = self.program._locals["color_nn"](tensor(color.noise()))
            result = argmax(prediction)

            if color.value[0] == result.item(): correct += 1

        return correct / len(colors)