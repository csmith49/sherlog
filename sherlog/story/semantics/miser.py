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

        dependencies : Iterable[Miser] (default=())

        distribution : Optional[Distribution]

        forced : bool (default=False)
        """
        self.value = value
        self.distribution = distribution
        self.forced = forced
        
        # we'll add some extra checks instead of revealing dependencies outright
        self._dependencies = dependencies
        

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
        """Computes the log-probability of the forcing.

        If the value is not forced, assume the 'forcing' always holds (hence the forced log-prob is 0).

        See also `Miser.log_prob`.

        Returns
        -------
        Tensor
        """
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

def forcing_scale(values):
    """Computes the likelihood of the forcing on all provided values.

    Parameters
    ----------
    values : Iterable[Miser]

    Returns
    -------
    Tensor
    """
    tau = torch.stack([v.forced_log_prob for v in values]).sum(0)
    return torch.exp(tau)

# FUNCTOR OPS -----------------------------------------------------------

def wrap(obj, batches=1, **kwargs):
    """Wraps a value in the Miser functor.

    Parameters
    ----------
    obj : Any

    batches : int (default=1)

    kwargs : ignored

    Returns
    -------
    Miser
    """
    if torch.is_tensor(obj):
        value = obj
    else:
        value = torch.tensor(obj)
    # make sure we've expanded!
    return Miser(batch.expand(value, batches))

def fmap(callable, args, kwargs, **fmap_args):
    logger.info(f"Calling {callable} on {args}.")
    value = batch.batch_map(callable, *[arg.value for arg in args])
    return Miser(value, dependencies=args)

def random_factory(distribution, batches=1, forcing=None):
    """Builds a Miser builtin (F v -> F v) from a distribution class.

    Parameters
    ----------
    distribution : distribution.Distribution

    batches : int (default=1)

    forcing : Optional[Observation]

    Returns
    -------
    Functor builtin
    """
    def builtin(*args, **kwargs):
        parameters = [arg.value for arg in args]
        dist = distribution(*parameters)

        # check if the assignment is forced
        assignment = kwargs["assignment"]
        if forcing and assignment.target in forcing:
            # this is a fresh value, so we have to lift first
            value = wrap(forcing[assignment.target], batches=batches).value

            # give a warning if we're forcing a reparameterizable distribution
            if dist.has_rsample:
                logger.warning(f"Forcing a value from reparameterizable distribution {distribution.__name__}.")
            
            logger.info(f"Forcing for {assignment.target}: {value} ~ {distribution.__name__}{parameters}.")

            # build as normal, but mark as forced            
            return Miser(
                value,
                dependencies=list(args),
                distribution=dist,
                forced=True
            )
        
        # if the distribution supports reparameterization, use it
        if dist.has_rsample: value = dist.rsample()
        else: value = dist.sample()
    
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

def factory(samples, forcing=None):
    """Builds a Miser execution functor.

    Parameters
    ----------
    samples : int

    forcing : Optional[Observation]

    Returns
    -------
    Functor
    """
    # encode the samples parameter in the wrap function
    batched_wrap = partial(wrap, batches=samples)

    # construct the builtins with the appropriate forcing (if provided)
    batched_random_factory = partial(random_factory, batches=samples, forcing=forcing)
    
    builtins = {
        "beta" : batched_random_factory(Beta),
        "bernoulli" : batched_random_factory(Bernoulli),
        "normal" : batched_random_factory(Normal),
        "tensorize" : lift(_tensorize),
        "equal" : lift(_equal),
        "satisfy" : lift(_satisfy),
        "set" : lift(_set)
    }

    # use constructed operations
    return Functor(batched_wrap, fmap, builtins)

functor = factory