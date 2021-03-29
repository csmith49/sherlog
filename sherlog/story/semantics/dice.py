from ...engine import Functor
from ...logs import get
from . import batch
import torch
from torch.distributions import Bernoulli, Normal, Beta
from itertools import chain

logger = get("story.semantics.dice")

class DiCE:
    def __init__ (self, value, dependencies=(), distribution=None):
        """A value wrapped by the DiCE functor.

        Parameters
        ----------
        value : Tensor

        dependencies : Iterable[DiCE]

        distribution : Distribution
        """
        self.value = value
        # we'll add some extra checks instead of revealing dependencies outright
        self._dependencies = dependencies
        self.distribution = distribution

    @property
    def is_stochastic(self):
        """True if a distribution has been provided.

        Returns
        -------
        bool
        """
        if self.distribution:
            return True
        return False

    @property
    def log_prob(self):
        """Computes the log-probability of having sampled the value from the provided distribution.

        If no distribution is provided, assume the value is deterministic (hence the log-prob is 0).

        Returns
        -------
        Tensor
        """
        if self.distribution:
            return self.distribution.log_prob(self.value)
        else:
            return torch.zeros(batch.batches(self.value))

    def dependencies(self):
        """Recursively compute all dependencies.

        Returns
        -------
        Iterable[DiCE]
        """
        # using pre-order, so yield current node first
        yield self

        # track all nodes already seen to avoid duplicates
        seen = set()
        for dependency in self._dependencies:
            # iterate over all dependencies of dependencies
            for dice in dependency.dependencies():
                if not dice in seen:
                    yield dice
                seen.add(dice)

    def batch_values(self):
        yield from self.value.unbind()

def magic_box(values):
    """Computes the magic box of the provided values.

    Parameters
    ----------
    values : Iterable[DiCE]

    Returns
    -------
    Tensor
    """
    tau = torch.stack([v.log_prob for v in values]).sum(0)
    result = torch.exp(tau - tau.detach())

    # a check to ensure gradients are being propagated
    if result.grad_fn is None:
        logger.warning(f"Magic box of {values} has no gradient.")
    
    return result

def wrap(obj, batches=1, **kwargs):
    if torch.is_tensor(obj):
        value = obj
    else:
        value = torch.tensor(obj)

    return DiCE(batch.expand(value, batches))

def fmap(callable, args, kwargs, **fmap_args):
    logger.info(f"Calling {callable} on {args}.")
    value = batch.batch_map(callable, *[arg.value for arg in args])
    return DiCE(value, dependencies=args)

def random_factory(distribution):
    """Builds a DiCE builtin (F v -> F v) from a distribution class.

    Parameters
    ----------
    distribution : distribution.Distribution

    Returns
    -------
    Functor builtin
    """
    def builtin(*args, **kwargs):
        parameters = [arg.value for arg in args]
        dist = distribution(*parameters)
        try:
            value = dist.rsample()
        except:
            value = dist.sample()
        logger.info(f"Sampling: {value} ~ {distribution.__name__}{parameters}.")
        return DiCE(value, dependencies=list(args), distribution=dist)
    return builtin

def lift(callable):
    def builtin(*args, **kwargs):
        value = batch.batch_map(callable, *[arg.value for arg in args])
        return DiCE(value, dependencies=list(args))
    return builtin

def _tensorize(*args):
    return torch.tensor(list(args))

def _equal(v1, v2):
    if torch.equal(v1, v2):
        return torch.tensor(1.0)
    else:
        return torch.tensor(0.0)

def _satisfy(meet, avoid):
    return meet * (1 - avoid)

def _set(x):
    return x

builtins = {
    "beta" : random_factory(Beta),
    "bernoulli" : random_factory(Bernoulli),
    "normal" : random_factory(Normal),
    "tensorize" : lift(_tensorize),
    "equal" : lift(_equal),
    "satisfy" : lift(_satisfy),
    "set" : lift(_set)
}

functor = Functor(wrap, fmap, builtins)