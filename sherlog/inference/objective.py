import storch
from ..logs import get

logger = get("inference.objective")

class Objective:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def is_nan(self):
        return self.value._tensor.isnan().any()
    
    def __str__(self):
        return f"Obj<{self.name}, {self.value._tensor}>"