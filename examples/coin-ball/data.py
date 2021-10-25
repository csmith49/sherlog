from enum import Enum
from random import gauss, choices, choice
from colorsys import rgb_to_hsv, hsv_to_rgb
from sherlog.inference import FunctionalEmbedding

class Coin(Enum):
    TAILS = 0, (1.0, 0.0)
    HEADS = 1, (0.0, 1.0)

    def noise(self, scaling : float = 0.03):
        t, h = self.value[-1]
        t = max(0.0, min(1.0, gauss(t, scaling)))
        h = max(0.0, min(1.0, gauss(h, scaling)))
        return (t, h)

def random_coin():
    return choice([Coin.TAILS, Coin.HEADS])

class Color(Enum):
    RED = 0, (1.0, 0.0, 0.0)
    GREEN = 1, (0.0, 1.0, 0.0)
    BLUE = 2, (0.0, 0.0, 1.0)

    def noise(self, scaling : float = 0.03):
        """Add noise to the RGB representation of the color."""

        h, s, v = rgb_to_hsv(*self.value[-1])
        h = gauss(h, scaling) % 1.0
        s = max(0.0, min(1.0, gauss(s, scaling)))
        v = max(0.0, min(1.0, gauss(v, scaling)))
        return hsv_to_rgb(h, s, v)

def random_color():
    return choice([Color.RED, Color.GREEN, Color.BLUE])

def outcome(coin : Coin, ball1 : Color, ball2 : Color) -> bool:
    if (coin == Coin.HEADS) and (ball1 == Color.RED or ball2 == Color.RED):
        return True
    elif (ball1 == ball2):
        return True
    else:
        return False

def sample(coin_weights, urn_one_weights, urn_two_weights):
    coin = choices([Coin.TAILS, Coin.HEADS], coin_weights)[0]
    ball1 = choices([Color.RED, Color.BLUE], urn_one_weights)[0]
    ball2 = choices([Color.RED, Color.GREEN, Color.BLUE], urn_two_weights)[0]

    return (coin.noise(), ball1.noise(), ball2.noise(), outcome(coin, ball1, ball2))

embedder = FunctionalEmbedding(
    evidence=lambda p : "game(coin, ball1, ball2, win)" if p[-1] else "game(coin, ball1, ball2, loss)",
    parameters=lambda p: {
        "coin" : p[0],
        "ball1" : p[1],
        "ball2" : p[2]
    }
)