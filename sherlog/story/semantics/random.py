from ...engine import Functor

def wrap(obj, **kwargs):
    return []

def fmap(callable, args, kwargs, **fmap_args):
    for arg in args:
        yield from arg

def _random(*args, assignment=None, **kwargs):
    for arg in args:
        yield from arg
    yield assignment

builtins = {
    "beta" : _random,
    "bernoulli" : _random,
    "normal" : _random
}

functor = Functor(wrap, fmap, builtins)