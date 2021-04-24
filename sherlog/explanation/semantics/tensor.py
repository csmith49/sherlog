"""Simple forward semantics that evaluates a story over tensors."""

from ...engine import Functor
import torch as t
import torch.distributions as td

def wrap(obj, **kwargs):
    if isinstance(obj, (t.Tensor,)): return obj
    else: return t.tensor(obj)

def fmap(callable, args, kwargs, **fmap_args):
    return callable(*args, **kwargs)

def _beta(p, q, **kwargs):
    return td.Beta(p, q).rsample()

def _bernoulli(p, **kwargs):
    return td.Bernoulli(p).sample()

def _normal(m, s, **kwargs):
    return td.Normal(m, s).rsample()

def _tensorize(*args, **kwargs):
    return t.tensor(list(args))

def _equal(v1, v2, **kwargs):
    if t.equal(v1, v2):
        return t.tensor(1.0)
    else:
        return t.tensor(0.0)

def _satisfy(meet, avoid, **kwargs):
    return meet * (1 - avoid)

def _set(value, **kwargs):
    return value

builtins = {
    "beta" : _beta,
    "bernoulli" : _bernoulli,
    "normal" : _normal,
    "tensorize" : _tensorize,
    "equal" : _equal,
    "satisfy" : _satisfy,
    "set" : _set
}

functor = Functor(wrap, fmap, builtins)