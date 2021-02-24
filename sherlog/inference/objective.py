import storch
from ..logs import get

logger = get("inference.objective")

class Objective:
    def __init__(self, reference, store):
        self.reference = reference
        self.store = store

    @property
    def value(self):
        return self.store[self.reference]

    def maximize(self):
        logger.info(f"Maximizing {self.reference}: {self.value}")
        storch.add_cost(-1 * self.value, self.reference.name)

    def minimize(self):
        logger.info(f"Minimizing {self.reference}: {self.value}")
        storch.add_cost(self.value, self.reference.name)

    def __str__(self):
        return f"{self.reference}: {self.value}"