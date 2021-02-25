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

    @property
    def name(self):
        return self.reference.name

    def __str__(self):
        return f"<Objective {self.name}: {self.value}>"