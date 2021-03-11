from ..engine import Algebra
from torch import tensor, is_tensor, Tensor
import torch.distributions as dists
import storch
from storch.method import Reparameterization, ScoreFunction, Expect, GumbelSoftmax


def lift(obj):
    if isinstance(obj, (Tensor, storch.Tensor)): return obj
    else: return tensor(obj)

def unlift(obj): return obj

def _beta(p, q, target=None, method=Reparameterization, method_kwargs={}, **kwargs):
    dist = dists.Beta(p, q)
    return method(target.name, **method_kwargs)(dist)

def _bernoulli(p, target=None, method=GumbelSoftmax, method_kwargs={}, **kwargs):
    dist = dists.Bernoulli(probs=p)
    return method(target.name, initial_temperature=1.0, **method_kwargs)(dist)

def _normal(m, s, target=None, method=Reparameterization, method_kwargs={}, **kwargs):
    dist = dists.Normal(m, s)
    return method(target.name, **method_kwargs)(dist)

builtins = {
    "beta" : _beta,
    "bernoulli" : _bernoulli,
    "normal" : _normal
}

algebra = Algebra(
    lift,
    unlift,
    builtins
)