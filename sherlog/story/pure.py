import torch
import torch.distributions as dists

from ..engine import Algebra

def lift(obj):
    if isinstance(obj, (torch.Tensor,)): return obj
    else: return torch.tensor(obj)

def unlift(obj): return obj

def _beta(p, q, target=None, method_kwargs={}, **kwargs):
    dist = dists.Beta(p, q)
    return dist.rsample()

def _normal(m, s, target=None, method_kwargs={}, **kwargs):
    dist = dists.Normal(m, s)
    return dist.rsample()

def _bernoulli(p, target=None, method_kwargs={}, **kwargs):
    temp = torch.tensor(0.01)
    dist = dists.RelaxedBernoulli(temp, probs=p)
    return dist.rsample()

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