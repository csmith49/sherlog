from .semantics import Algebra, run_factory
from torch import tensor, is_tensor
import torch.distributions as dists

def tag(value):
    if not is_tensor(value): return tensor(value)
    else: return value

def untag(value): return value

def normal(mean, sdev, *args, target=None, method=None, method_args={}, **kwargs):
    dist = dists.Normal(mean, sdev)
    method = method(target, **method_args)
    return method(dist)

def beta(alpha, beta, *args, target=None, method=None, method_args={}, **kwargs):
    dist = dists.Beta(alpha, beta)
    method = method(target, **method_args)
    return method(dist)

def bernoulli(prob, *args, target=None, method=None, method_args={}, **kwargs):
    dist = dists.Bernoulli(prob)
    method = method(target, **method_args)
    return method(dist)

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