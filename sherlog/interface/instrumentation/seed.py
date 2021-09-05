from hashids import Hashids
from random import randint
from typing import Optional

# initialize hashid conversion process
hashids = Hashids()

# seeds are human-readable random values
class Seed:
    """Human-readable random value."""

    def __init__(self, value : Optional[int] = None):
        """Construct a seed."""

        value = value if value else randint(0, 100000)
        self.value = hashids.encode(value)

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"Seed({self.value})"