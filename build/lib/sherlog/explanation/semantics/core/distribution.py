from torch.distributions import Distribution
from torch import Tensor
from typing import List
import torch.distributions as dists

from typing import Set

# DISTRIBUTION UTILITIES

_DISTRIBUTION_MAP = {
    "bernoulli" : dists.Bernoulli,
    "categorical" : dists.Categorical,
    "normal" : dists.Normal,
    "beta" : dists.Beta
}

def supported_distributions() -> Set[str]:
    return set(_DISTRIBUTION_MAP.keys())

def lookup_constructor(name : str):
    return _DISTRIBUTION_MAP[name]
