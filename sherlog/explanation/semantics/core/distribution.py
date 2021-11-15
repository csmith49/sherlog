from torch import Tensor, softmax, ones
import torch.distributions as dists

from typing import Set

def discrete_constructor(dimension : Tensor):
    parameters = softmax(ones(dimension), dim=0)
    return dists.Categorical(parameters)

# DISTRIBUTION UTILITIES

_DISTRIBUTION_MAP = {
    "bernoulli" : dists.Bernoulli,
    "categorical" : dists.Categorical,
    "normal" : dists.Normal,
    "beta" : dists.Beta,
    "dirichlet" : dists.Dirichlet,
    "discrete" : discrete_constructor
}

def supported_distributions() -> Set[str]:
    return set(_DISTRIBUTION_MAP.keys())

def lookup_constructor(name : str):
    return _DISTRIBUTION_MAP[name]
