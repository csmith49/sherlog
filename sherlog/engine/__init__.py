"""Engines provide three primary pieces of information:
1. What to do with constants,
2. what to do with built-in functions, and
3. how to apply externally-provided functions.

Functors provide the necessary structure. Let `T` be an execution context; functors are defined with:
1. `wrap` is a function of type `* -> T`
2. `fmap` is a function of type `(* -> *) -> * -> T`
"""

from .model import Model
from .store import Store
from .functor import Functor
from .assignment import Assignment
from .value import Value, Identifier, Literal