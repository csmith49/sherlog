"""An engine requires three pieces of information:
1. What to do with constants,
2. what to do with built-in functions, and
3. how to apply external functions.
"""

from .model import Model
from .store import Store
from .functor import Functor
from . import value