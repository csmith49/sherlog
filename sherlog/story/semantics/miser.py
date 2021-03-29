from ...engine import Functor
from ...logs import get
from . import batch
import torch
from torch.distributions import Bernoulli, Normal, Beta
from itertools import chain
from functools import partial

logger = get("story.semantics.miser")

class Miser:
    def __init__ (self, value, dependencies=(), distribution=None, forced=False):
        """A value wrapped by the Miser functor.

        Parameters
        ----------
        value : Tensor

        dependencies : Iterable[Miser]

        distribution : Distribution

        forced : bool
        """
        self.value = value
        # we'll add some extra checks instead of revealing dependencies outright
        self._dependencies = dependencies
        self.distribution = distribution
        self.forced = forced

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
    def forced_log_prob(self):
        if self.distribution and self.forced:
            return self.distribution.log_prob(self.value)
        else:
            return torch.zeros(batch.batches(self.value))

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
        Iterable[Miser]
        """
        # using pre-order, so yield current node first
        yield self

        # track all nodes already seen to avoid duplicates
        seen = set()
        for dependency in self._dependencies:
            # iterate over all dependencies of dependencies
            for miser in dependency.dependencies():
                if not miser in seen:
                    yield miser
                seen.add(miser)

    def batch_values(self):
        yield from self.value.unbind()

def magic_box(values):
    """Computes the magic box of the provided values.

    Parameters
    ----------
    values : Iterable[Miser]

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

def sample_scale(values, forcing):
    tau = torch.stack([v.forced_log_prob for v in values]).sum(0)
    return torch.exp(tau)

def wrap(obj, batches=1, **kwargs):
    if torch.is_tensor(obj):
        value = obj
    else:
        value = torch.tensor(obj)

    return Miser(batch.expand(value, batches))

def fmap(callable, args, kwargs, **fmap_args):
    logger.info(f"Calling {callable} on {args}.")
    value = batch.batch_map(callable, *[arg.value for arg in args])
    return Miser(value, dependencies=args)

def random_factory(distribution, forcing={}):
    """Builds a Miser builtin (F v -> F v) from a distribution class.

    Parameters
    ----------
    distribution : distribution.Distribution

    forcing : Dict[str, Tensor]

    Returns
    -------
    Functor builtin
    """
    def builtin(*args, **kwargs):
        parameters = [arg.value for arg in args]
        dist = distribution(*parameters)

        # check if the assignment is forced
        assignment = kwargs["assignment"]
        try:
            value = forcing[assignment.target.name].value
            return Miser(value, dependencies=list(args), distribution=dist, forced=True)
        except: pass
        
        # build the value normally
        try:
            value = dist.rsample()
        except:
            value = dist.sample()
    
        logger.info(f"Sampling: {value} ~ {distribution.__name__}{parameters}.")
        return Miser(value, dependencies=list(args), distribution=dist)
    return builtin

# DETERMINISTIC BUILTINS ---------------------------------------------------------

def lift(callable):
    def builtin(*args, **kwargs):
        value = batch.batch_map(callable, *[arg.value for arg in args])
        return Miser(value, dependencies=list(args))
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

# FACTORY ------------------------------------------------------------------------

def factory(samples, forcing={}):
    """Builds a Miser execution functor.

    Parameters
    ----------
    samples : int

    forcing : Dict[str, Tensor]

    Returns
    -------
    Functor
    """
    # encode the samples parameter in the wrap function
    batched_wrap = partial(wrap, batches=samples)

    # construct the builtins with the appropriate forcing
    wrapped_forcing = {k : batched_wrap(v) for k, v in forcing.items()}
    forced_factory = partial(random_factory, forcing=wrapped_forcing)
    
    builtins = {
        "beta" : forced_factory(Beta),
        "bernoulli" : forced_factory(Bernoulli),
        "normal" : forced_factory(Normal),
        "tensorize" : lift(_tensorize),
        "equal" : lift(_equal),
        "satisfy" : lift(_satisfy),
        "set" : lift(_set)
    }

    # use constructed operations
    return Functor(batched_wrap, fmap, builtins)

functor = factory