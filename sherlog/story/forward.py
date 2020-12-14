from ..engine import factory
from torch import tensor, is_tensor
import torch.distributions as dists

def lift(obj):
    if not is_tensor(obj): return tensor(obj)
    else: return obj

def unlift(obj): return obj

def _beta(p, q, **kwargs):
    dist = dists.Beta(p, q)
    return dist.rsample()

def _bernoulli(p, **kwargs):
    dist = dists.Bernoulli(p)
    return dist.sample()

def _normal(m, s, **kwargs):
    dist = dists.Normal(m, s)
    return dist.rsample()

builtins = {
    "beta" : _beta,
    "bernoulli" : _bernoulli,
    "normal" : _normal
}

algebra = factory(
    lift,
    unlift,
    builtins
)