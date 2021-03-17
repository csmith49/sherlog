from ...engine import Functor
import torch as t
import torch.distributions as td
from itertools import chain

class DiCE:
    def __init__(self, value, log_prob=None, dependencies=None):
        self.value = value
        self.log_prob = log_prob
        self._dependencies = dependencies

    @property
    def is_stochastic(self):
        if self.log_prob:
            return True
        return False

    # TODO - filter by unique deps
    def dependencies(self):
        yield self
        if self._dependencies:
            for dep in self._dependencies:
                yield from dep.dependencies()

def magic_box(values):
    tau = t.tensor(0.0)
    for v in values:
        if v.is_stochastic:
            tau += v.log_prob
    return t.exp(tau - tau.detach())

def wrap(obj, **kwargs):
    if t.is_tensor(obj):
        value = obj
    else:
        value = t.tensor(obj)
    return DiCE(value)

def fmap(callable, args, kwargs, **fmap_args):
    value = callable(*[arg.value for arg in args], **kwargs)
    return DiCE(value, dependencies=args)

def _beta(p, q, **kwargs):
    dist = td.Beta(p.value, q.value)
    value = dist.rsample()
    return DiCE(value, log_prob=dist.log_prob(value), dependencies=[p, q])

def _normal(m, s, **kwargs):
    dist = td.Normal(m.value, s.value)
    value = dist.rsample()
    return DiCE(value, log_prob=dist.log_prob(value), dependencies=[m, s])

def _bernoulli(p, **kwargs):
    dist = td.Bernoulli(p.value)
    value = dist.sample()
    return DiCE(value, log_prob=dist.log_prob(value), dependencies=[p])

def _random(distribution, *args, **kwargs):
    parameters = [arg.value for arg in args]
    dist = distribution(*parameters)
    try:
        value = dist.rsample()
    except:
        value = dist.sample()
    log_prob = dist.log_prob(value)
    return DiCE(value, log_prob=log_prob, dependencies=list(args))

def _tensorize(*args, **kwargs):
    value = t.tensor([arg.value for arg in args])
    return DiCE(value, dependencies=list(args))

def _equal(v1, v2, **kwargs):
    if t.equal(v1.value, v2.value):
        value = t.tensor(1.0)
    else:
        value = t.tensor(0.0)
    return DiCE(value, dependencies=[v1, v2])

def _satisfy(meet, avoid, **kwargs):
    value = meet.value * (1 - avoid.value)
    return DiCE(value, dependencies=[meet, avoid])

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