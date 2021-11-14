from torch import tensor, Tensor, stack, dist
import torch.distributions as tdist
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

def gaussian_survival(a, mu, sigma):
    return tdist.Normal(mu, sigma).cdf(a)

# FOR EVALUATING LITERALS

def _literal_equal(left : Tensor, right : Tensor) -> Tensor:
    if dist(left.float(), right.float()) <= 1e-6:
        return tensor(1.0)
    else:
        return tensor(0.0)

def _literal_not_equal(left : Tensor, right : Tensor) -> Tensor:
    if dist(left.float(), right.float()) > 1e-6:
        return tensor(1.0)
    else:
        return tensor(0.0)


# FOR EVALUATING OBSERVATIONS

def _observation_product(*args : Tensor) -> Tensor:
    return stack(args).prod()

def _observation_sum(*args : Tensor) -> Tensor:
    return stack(args).sum()

# BUILTIN UTILITIES

_BUILTIN_MAP = {
    "tensorize" : tensorize,
    "identity" : identity,
    "or" : _or,
    "gt" : gt,
    "add" : add,
    "max" : max,
    "gaussian_survival" : gaussian_survival,
    # LITERALS
    "equal" : _literal_equal,
    "not equal" : _literal_not_equal,
    # OBSERVATIONS
    "product" : _observation_product,
    "sum" : _observation_sum
}

def supported_builtins():
    return set(_BUILTIN_MAP.keys())

def lookup(name : str) -> Callable[..., Tensor]:
    return _BUILTIN_MAP[name]
