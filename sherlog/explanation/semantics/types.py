from ...engine import Functor
from ...logs import get

logger = get("story.semantics.types")

# DEFINING THE TYPES

# Note - our typing is primarily used for *forcing*, so many of the typical constructions
# are unnecessary. In the future, we may extend this implementation with, e.g.:
# 1. Subtyping
# 2. Type-checking
# 3. Further domain refinement, such as positive reals and unit intervals

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Type: pass

# the only type with a payload
@dataclass(frozen=True)
class Discrete(Type):
    cardinality: Optional[int] = None

@dataclass(frozen=True)
class Real(Type): pass

@dataclass(frozen=True)
class Any(Type): pass

# FUNCTOR SEMANTICS

def wrap(obj, **kwargs):
    return Any()

def fmap(callable, args, kwargs, **fmap_args):
    return Any()

def random_factory(distribution):
    """Builds a Type builtin (T _ -> T _) from a distribution class.

    Parameters
    ----------
    distribution : distribution.Distribution

    Returns
    -------
    Functor builtin
    """
    # manually encoding the output
    output_type = {
        "Bernoulli" : Discrete(2),
        "Normal" : Real(),
        "Beta" : Real()
    }[distribution.__name__]

    # the trivial warpped function
    def builtin(*args, **kwargs):
        return output_type
    return builtin        

def lifted(*args, **kwargs):
    return Any()

# PUT IT TOGETHER

from collections import defaultdict
from torch.distributions import Beta, Normal, Bernoulli

builtins = defaultdict(lambda: lifted)
builtins["beta"]      = random_factory(     Beta)
builtins["normal"]    = random_factory(   Normal)
builtins["bernoulli"] = random_factory(Bernoulli)

functor = Functor(wrap, fmap, builtins)