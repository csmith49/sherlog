from .semantics import Algebra, run_factory
from torch import tensor, is_tensor
import pyro
import pyro.distributions as dists

def tag(value):
    if not is_tensor(value): return tensor(value)
    else: return value

def untag(value): return value

def normal(mean, sdev, *args, target=None, **kswarg):
    dist = dists.Normal(mean, sdev)
    return pyro.sample(target, dist)

def beta(alpha, beta, *args, target=None, **kwargs):
    dist = dists.Beta(alpha, beta)
    return pyro.sample(target, dist)

def bernoulli(prob, *args, target=None, **kwargs):
    dist = dists.Bernoulli(prob)
    return pyro.sample(target, dist)

algebra = Algebra(
    tag,
    untag,
    {
        "normal" : normal,
        "beta" : beta,
        "bernoulli" : bernoulli
    }
)
run = run_factory(algebra)
