from torch import tensor, Tensor, stack
from typing import Callable

# DETERMINISTIC BUILTINS

def tensorize(*args : Tensor) -> Tensor:
    return stack(args) # stack maintains grads, other options like cat do not

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

def max(*args):
    return stack(args).max()

# BUILTIN UTILITIES

_BUILTIN_MAP = {
    "tensorize" : tensorize,
    "identity" : identity,
    "or" : _or,
    "gt" : gt,
    "add" : add,
    "max" : max
}

def supported_builtins():
    return set(_BUILTIN_MAP.keys())

def lookup(name : str) -> Callable[..., Tensor]:
    return _BUILTIN_MAP[name]