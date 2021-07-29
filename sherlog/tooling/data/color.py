from colorsys import rgb_to_hsv, hsv_to_rgb
from ...logs import get
from ..evaluation.datasource import DataSource
from random import gauss, choice
from enum import Enum

logger = get("tooling.data.color")

BaseColor = Enum("BaseColor", "RED GREEN BLUE")
base_color_values = {
    BaseColor.RED : (1.0, 0.0, 0.0),
    BaseColor.GREEN : (0.0, 1.0, 0.0),
    BaseColor.BLUE : (0.0, 0.0, 1.0)
}

class Color:
    def __init__(self, base, red, green, blue):
        self._base = base
        self._red, self._green, self._blue = red, green, blue

class ColorDataSource(DataSource):
    def __init__(self, scaling=0.03):
        self._scaling = scaling

    def _noisy_color_values(self, color):
        hue, saturation, value = rgb_to_hsv(*base_color_values[color])
        hue = gauss(hue, self._scaling) % 1.0
        saturation = max(0.0, min(1.0, gauss(saturation, self._scaling)))
        value = max(0.0, min(1.0, gauss(value, self._scaling)))
        return hsv_to_rgb(hue, saturation, value)

    def _sample(self):
        color = choice(base_color_values.keys())
        r, g, b = self._noisy_color_values(color)
        return Color(color, r, g, b)
        
    def training_data(self, *args, quantity=1, **kwargs):
        for _ in range(quantity):
            yield self._sample()

    def testing_data(self, *args, quantity=1, **kwargs):
        for _ in range(quantity):
            yield self._sample()
