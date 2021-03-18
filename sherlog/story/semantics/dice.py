from ...engine import Functor
from ...logs import get
import torch
from torch.distributions import Bernoulli, Normal, Beta
from itertools import chain

logger = get("story.semantics.dice")

class DiCE:
    def __init__(self, value, log_prob=None, dependencies=None):
        """A value wrapped by the DiCE functor.

        Parameters
        ----------
        value : Tensor

        log_prob : Optional[Tensor]

        dependencies : Iterable[DiCE]
        """
        self.value = value
        self.log_prob = log_prob
        self._dependencies = dependencies

    @property
    def is_stochastic(self):
        """True if a log-probability is defined.

        Returns
        -------
        bool
        """
        if self.log_prob:
            return True
        return False

    # TODO - filter by unique deps
    def dependencies(self):
        """Recursively computes all dependencies.

        Returns
        -------
        Iterable[DiCE]
        """
        yield self
        if self._dependencies:
            for dep in self._dependencies:
                yield from dep.dependencies()

def magic_box(*args):
    """Computes the magic box of the provided values.

    Parameters
    ----------
    *args : Iterable[DiCE]

    Returns
    -------
    Tensor
    """
    tau = torch.tensor(0.0)
    for dice in args:
        if dice.is_stochastic:
            tau += dice.log_prob
    result = torch.exp(tau - tau.detach())
    
    # a quick check to ensure we're getting gradients
    if result.grad_fn is None:
        logger.warning(f"Magic box of {values} has no gradient.")

    return result

def wrap(obj, **kwargs):
    if torch.is_tensor(obj):
        value = obj
    else:
        value = torch.tensor(obj)
    return DiCE(value)

def fmap(callable, args, kwargs, **fmap_args):
    logger.info(f"Calling {callable} on {args}.")
    value = callable(*[arg.value for arg in args], **kwargs)
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
        dist = distribution(*[arg.value for arg in args])
        try:
            value = dist.rsample()
        except:
            value = dist.sample()
        log_prob = dist.log_prob(value)
        logger.info(f"Sampling: {value} ~ {distribution.__name__}{parameters} with log-prob {log_prob}.")
        return DiCE(value, log_prob=log_prob, dependencies=list(args))
    return builtin

def lift(callable):
    def builtin(*args, **kwargs):
        value = callable(*[arg.value for arg in args])
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

builtins = {
    "beta" : random_factory(Beta),
    "bernoulli" : random_factory(Bernoulli),
    "normal" : random_factory(Normal),
    "tensorize" : lift(_tensorize),
    "equal" : lift(_equal),
    "satisfy" : lift(_satisfy),
    "set" : lift(lambda x: x)
}

functor = Functor(wrap, fmap, builtins)