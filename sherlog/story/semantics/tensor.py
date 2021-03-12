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

builtins = {
    "beta" : _beta,
    "bernoulli" : _bernoulli,
    "normal" : _normal
}

functor = Functor(wrap, fmap, builtins)