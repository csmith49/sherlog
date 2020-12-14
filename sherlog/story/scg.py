from ..engine import factory
from torch import tensor, is_tensor, Tensor
import torch.distributions as dists
import storch

def lift(obj):
    if isinstance(obj, (Tensor, storch.Tensor)): return obj
    else: return tensor(obj)

def unlift(obj): return obj

def _beta(p, q, target=None, method=storch.method.Reparameterization, method_kwargs={}, **kwargs):
    dist = dists.Beta(p, q)
    return method(target.name, **method_kwargs)(dist)

def _bernoulli(p, target=None, method=storch.method.ScoreFunction, method_kwargs={}, **kwargs):
    dist = dists.Bernoulli(p)
    return method(target.name, **method_kwargs)(dist)

def _normal(m, s, target=None, method=storch.method.Reparameterization, method_kwargs={}, **kwargs):
    dist = dists.Normal(m, s)
    return method(target.name, **method_kwargs)(dist)

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