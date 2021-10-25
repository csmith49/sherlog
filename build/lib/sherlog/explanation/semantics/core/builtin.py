from torch import tensor, Tensor, exp
from storch import cat
from typing import Callable

# DETERMINISTIC BUILTINS

def identity(value : Tensor) -> Tensor:
    return value

def _or(*args):
    prod = tensor(1.0)
    for arg in args:
        prod *= (1 - arg)
    return 1 - prod

def gt(left : Tensor, right : Tensor) -> Tensor:
    return tensor(1.0) if left >= right else tensor(0.0)

def add(left : Tensor, right : Tensor) -> Tensor:
    return left + right

def _max(*args):
    # approximated using logsumexp
    total = tensor(0.0)
    for argument in args:
        total = total + exp(argument)
    return total.log()

# OBSERVATION EVALUATION BUILTINS

def dimension_magnitude(left : Tensor, right : Tensor) -> Tensor:
    return (left - right) ** 2

def equality_ball(*arguments : Tensor) -> Tensor:
    # convert l2 distance to 0-1 (ish) values with Gaussain RBF
    total = sum(arguments)
    return exp(-(total ** 2))

# BUILTIN UTILITIES

_BUILTIN_MAP = {
    "identity" : identity,
    "or" : _or,
    "gt" : gt,
    "add" : add,
    "max" : _max,
    "dimension_magnitude" : dimension_magnitude,
    "equality_ball" : equality_ball
}

def supported_builtins():
    return set(_BUILTIN_MAP.keys())

def lookup(name : str) -> Callable[..., Tensor]:
    return _BUILTIN_MAP[name]