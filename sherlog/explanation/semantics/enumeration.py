"""Functor for identifying enumeration sites."""

from typing import Callable, List, Optional, Dict, Tuple, Iterable
from torch import Tensor, zeros, tensor
from torch.distributions import Categorical, Bernoulli
from itertools import product, chain
from collections import defaultdict

from ...engine import Functor, Assignment, Value, Literal, Identifier
from ...logs import get

logger = get("explanation.semantics.enumeration")

# Type Annotations

# possibilities are the functor-value here
Support = List[Value]
Possibility = List[Tuple[Identifier, Value]]
Enumerator = Callable[[Assignment], Optional[Possibility]]

# Utility

def arg_lists(possibility : Possibility) -> Iterable[Dict[str, Value]]:
    """Construct all possible argument lists from a possibility.

    Parameters
    ----------
    possibility : Possibility

    Yields
    ------
    Dict[str, Value]
    """
    keys, possibilities = zip(*possibility)
    for possibility in product(*possibilities):
        yield dict(zip(keys, possibility))

# Built-In Enumerators

def natural_enumerator(assignment : Assignment) -> Optional[Support]:
    """
    
    """
    return [Literal(i) for i in range(10)]

def categorical_enumerator(assignment : Assignment) -> Optional[Support]:
    """Sets assignment support to be the natural numbers in `[0, d]`, where `d` is the dimension of the inputs to `assignment`.

    Parameters
    ----------
    assignment : Assignment

    Returns
    -------
    Optional[Support]
    """
    # find the number of parameters
    dim = len(list(assignment.dependencies()))

    # build the range
    return [Literal(i) for i in range(dim)]

def bernoulli_enumerator(_ : Assignment) -> Optional[Support]:
    """Sets assignment support to be `{0, 1}`.
    
    Parameters
    ----------
    _ : Assignment
        Unused.

    Returns
    -------
    Optional[Support]
    """
    return [Literal(0), Literal(1)]

# Functor Construction

def wrap(obj, **kwargs):
    return []

def fmap(callable, args, kwargs, **fmap_args):
    return list(chain(**args))

def builtin_factory(enumerator=lambda _: None):
    """Constructs a built-in that attempts to apply `enumerator` to the given assignment.

    Parameters
    ----------
    enumerator : Optional[Enumerator]
        If no enumeration is given, the built-in just acts like `fmap` of an arbitrary function.
    
    Returns
    -------
    Functor builtin
    """
    def builtin(*args, **kwargs):
        # unpack the assignment and apply the pred
        assignment = kwargs['assignment']
        domain = enumerator(assignment)
        target = assignment.target

        # if the enumerator gives useful info, short-circuit
        if domain:
            result = [(target, domain)]
            logger.info(f"Found enumeration site at assignment {assignment}: support of {domain}")
            return list(chain(result, *args))
        
        # if we never short-circuit, default to no enumeration
        return list(chain(*args))
    return builtin

# Functor Factory

# default enumerators used by the factory - can be overwritten on construction
default_enumerators = {
    'categorical' : natural_enumerator,
    'bernoulli' : bernoulli_enumerator
}

def factory(enumerators : Optional[Dict[str, Enumerator]] = None) -> Functor[Possibility]:
    """Construct an enumeration functor.
    
    Parameters
    ----------
    enumerators : Optional[Dict[str, Enumerator]]
        If not provided, default enumerators are used instead.

    Returns
    -------
    Functor[Possibility]
    """

    # get the enumerators to be used
    enums = enumerators if enumerators is not None else default_enumerators

    # construct builtins
    builtins = defaultdict(lambda: builtin_factory())
    for id, enum in enums.items():
        builtins[id] = builtin_factory(enumerator=enum)

    return Functor(wrap, fmap, builtins)
