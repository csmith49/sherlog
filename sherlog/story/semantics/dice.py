from ...engine import Functor
import torch as t
import torch.distributions as td
from itertools import chain

# samples hold generated values with their source distribution
class Sample:
    def __init__(self, value, distribution):
        self.value = value
        self.distribution = distribution
    
    def log_prob(self):
        return self.distribution.log_prob(self.value)

# magic box operator for a set of samples
def magic_box(samples):
    tau = t.sum([s.log_prob() for s in samples])
    return t.exp(t - t.detach())

# unwraps the generated store
def unwrap(store):
    values, samples = {}, {}
    for k, v in store.items():
        value, sample = v
        values[k] = value
        samples[k] = sample
    return values, samples

# FUNCTOR SEMANTICS

def wrap(obj, **kwargs):
    # wrap the value
    if isinstance(obj, (t.Tensor,)):
        value = obj
    else:
        value = t.tensor(obj)
    
    # return paired with empty generator
    return (value, [])

def fmap(callable, args, kwargs, **fmap_args):
    args, samples = zip(*args)
    value = callable(*args, **kwargs)
    return (value, chain.from_iterable(samples))

def _beta(p, q, **kwargs):
    dist = td.Beta(p[0], q[0])
    value = dist.rsample()
    return (value, chain([Sample(value, dist)], p[1], q[1]))

def _bernoulli(p, **kwargs):
    dist = td.Bernoulli(p[0])
    value = dist.sample()
    return (value, chain([Sample(value, dist)], p[1]))

def _normal(m, s, **kwargs):
    dist = td.Normal(m[0], s[0])
    value = dist.rsample()
    return (value, chain([Sample(value, dist)], m[1], s[1]))

builtins = {
    "beta" : _beta,
    "bernoulli" : _bernoulli,
    "normal" : _normal
}

functor = Functor(wrap, fmap, builtins)