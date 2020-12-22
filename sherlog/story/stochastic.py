from ..engine import Algebra
from torch import tensor, is_tensor
import pyro.distributions as dists
from pyro.infer.predictive import Predictive
import pyro

def delta(target, value):
    return pyro.deterministic(target.name, value)

def lift(obj):
    if not is_tensor(obj): return tensor(obj)
    else: return obj

def unlift(obj): return obj

def _beta(p, q, target=None, **kwargs):
    dist = dists.Beta(p, q)
    return pyro.sample(target.name, dist)

def _bernoulli(p, target=None, **kwargs):
    dist = dists.Bernoulli(probs=p)
    return pyro.sample(target.name, dist)

def _normal(m, s, target=None, **kwargs):
    dist = dists.Normal(m, s)
    return pyro.sample(target.name, dist)

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