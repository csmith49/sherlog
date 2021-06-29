"""Miser gradient estimation functor. Based on DiCE."""

from functools import partial
from torch.distributions import Bernoulli, Normal, Beta, Categorical, Dirichlet
from typing import Optional, Iterable, Any
from torch import Tensor, tensor, stack
from torch.distributions import Distribution
import torch

from .target import Target
from ..observation import Observation
from ...engine import Functor
from ...logs import get

logger = get("explanation.semantics.miser")

class Miser:
    """Miser values maintain local dependencies and forcing behavior."""

    def __init__ (self,
        value : Tensor,
        dependencies : Iterable["Miser"] = (),
        distribution : Optional[Distribution] = None,
        forced : bool = False
    ):
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
    def is_stochastic(self) -> bool:
        """True if a distribution has been provided.

        Returns
        -------
        bool
        """
        return self.distribution is not None

    @property
    def is_forced(self) -> bool:
        """True iff the value has been forced.

        Returns
        -------
        bool
        """
        return self.forced

    def log_prob(self) -> Tensor:
        """The log-probability of having sampled the value from the provided distribution.

        If the value is not stochastic, the returned log-prob is 0.

        Returns
        -------
        Tensor
        """
        return self.distribution.log_prob(self.value) if self.is_stochastic else tensor(0.0)

    def dependencies(self) -> Iterable["Miser"]:
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

    def __repr__(self) -> str:
        return repr(self.value)

    @property
    def surrogate(self) -> Tensor:
        """Surrogate value for optimization.

        Returns
        -------
        Tensor
        """
        # step 1: compute forcing likelihood
        forced_dependencies = filter(lambda v: v.is_forced, self.dependencies())
        forcing_likelihood = stack([v.log_prob() for v in forced_dependencies]).sum().exp()

        # step 2: compute magic box of dependencies
        tau = stack([v.log_prob() for v in self.dependencies()]).sum()
        magic_box = (tau - tau.detach()).exp()

        # step 3: put it all together
        return self.value * magic_box * forcing_likelihood

# FUNCTOR OPS -----------------------------------------------------------

def wrap(obj : Any, **kwargs) -> Miser:
    """Wraps a value in the Miser functor.

    Parameters
    ----------
    obj : Any

    kwargs : ignored

    Returns
    -------
    Miser
    """
    # convert to tensor, if needed
    if torch.is_tensor(obj):
        value = obj
    else:
        value = torch.tensor(obj)
    
    # make sure we've fully wrapped
    return Miser(value)

def fmap(callable, args : Iterable[Miser], kwargs, **fmap_args) -> Miser:
    """
    Parameters
    ----------
    callable : Callable[]
    args
    kwargs

    **fmap_args

    Returns
    -------
    Miser
    """
    logger.info(f"Calling {callable} on {args}.")
    value = callable(*[arg.value for arg in args])
    logger.info(f"Call to {callable} produced result {value}.")
    return Miser(value, dependencies=args)

def distribution_factory(distribution : Distribution, forcing : Optional[Observation] = None):
    """Builds a Miser builtin (F v -> F v) from a distribution class.

    Parameters
    ----------
    distribution : distribution.Distribution

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
            value = wrap(forcing[assignment.target]).value

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

def builtin_factory(callable):
    """Lifts a callable to operate over MIser values.

    Parameters
    ----------
    callable : Tensors -> Tensor

    Returns
    -------
    Functor builtin
    """
    def builtin(*args, **kwargs):
        value = callable(*[arg.value for arg in args])
        return Miser(value, dependencies=list(args))
    return builtin

# DETERMINISTIC BUILTINS ---------------------------------------------------------

def _tensorize(*args):
    return torch.stack(args) # stack maintains gradients, tensor construction doesn't

def _or(*args):
    prod = torch.tensor(1.0)
    for arg in args:
        prod *= (1 - arg)
    return 1 - prod

def _set(x):
    return x

# FACTORY ------------------------------------------------------------------------

def factory(target : Target, forcing : Optional[Observation] = None) -> Functor:
    """Builds a Miser execution functor.

    Parameters
    ----------
    target : Target

    forcing : Optional[Observation]

    Returns
    -------
    Functor
    """
    # construct the builtins with the appropriate forcing (if provided)    
    builtins = {
        "beta" : distribution_factory(Beta, forcing=forcing),
        "bernoulli" : distribution_factory(Bernoulli, forcing=forcing),
        "normal" : distribution_factory(Normal, forcing=forcing),
        "categorical" : distribution_factory(Categorical, forcing=forcing),
        "dirichlet" : distribution_factory(Dirichlet, forcing=forcing),
        "tensorize" : builtin_factory(_tensorize),
        "target" : builtin_factory(target),
        "set" : builtin_factory(_set),
        "or" : builtin_factory(_or)
    }

    # use constructed operations
    return Functor(wrap, fmap, builtins)