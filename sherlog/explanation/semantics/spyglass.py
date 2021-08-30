from torch.distributions import Distribution
from sherlog.explanation.semantics.core.distribution import supported_distributions
from typing import Callable, List, Mapping, Iterable, Optional
from torch import Tensor, tensor, stack
from functools import partial
from itertools import unique, chain

from ...pipe import DynamicNamespace, Semantics, Monad, Statement

from .core.target import Target
from . import core

# MONADIC SEMANTICS

class Clue:
    """Monadic constructor for the Spyglass algorithm."""

    def __init__(self,
        value : Tensor,
        dependencies : Iterable['Clue'] = (),
        distribution : Optional[Distribution] = None,
        forced : bool = False
    ):
        """Construct a clue."""

        self.value = value
        self._dependencies = list(dependencies) # we'll wrap this
        self.distribution = distribution
        self.forced = forced

    def conditional_log_prob(self) -> Tensor:
        """Log-prob of `value` conditioned on the dependencies."""

        if self.distribution:
            return self.distribution.log_prob(self.value)
        else:
            return tensor(0.0)

    def forcing_log_prob(self) -> Tensor:
        """Likelihood of producing forced value."""

        if self.forced:
            return self.conditional_log_prob()
        else:
            return tensor(0.0)

    @property
    def dependencies(self) -> Iterable['Clue']:
        """Iterates over all dependencies in pre-order."""

        all_deps = chain.from_iterable(dep.dependencies for dep in self._dependencies)
        yield self # pre-order
        yield from unique(all_deps)

    @property
    def surrogate(self) -> Tensor:
        """Surrogate optimization value."""

        # default to a likelihood of 1
        mb, forcing = [tensor(0.0)], [tensor(0.0)]

        # construct magic box and forcing scaling values in one pass
        for dependency in self.dependencies:
            mb.append(dependency.conditional_log_prob())
            forcing.append(dependency.forcing_log_prob())

        forcing_likelihood = stack(forcing).sum().exp()

        tau = stack(mb).sum()
        magic_box = (tau - tau.detach()).exp()

        return self.value * magic_box * forcing_likelihood

def unit(value : Tensor) -> Clue:
    """Wrap a value in a clue constructor."""

    return Clue(value)

def bind(callable : Callable[..., Clue], arguments : List[Clue]) -> Clue:
    """Evaluate a monadic callable on a list of monadic arguments."""

    result = callable(*(argument.value for argument in arguments))
    result.dependencies = arguments # never set by callable
    return result

# NAMESPACE CONSTRUCTION

def lift(callable : Callable) -> Callable[..., Clue]:
    """Lift a builtin identifier to a monadic function."""

    def wrapped(*args):
        result = callable(*args)
        return unit(result)

    return wrapped

def lift_distribution(name : str) -> Callable[..., Clue]:
    """Lift a distribution identifier to a monadic function."""

    def wrapped(*args):
        distribution = core.distribution.lookup_constructor(name)(*args)
        value = distribution.rsample() if distribution.has_rsample() else distribution.sample()
        return Clue(value=value, distribution=distribution)
    
    return wrapped

def lift_forcing(name : str, forced_value : Tensor) -> Callable[..., Clue]:
    """Lift a distribution identifier and forced value to a monadic function."""

    def wrapped(*args):
        distribution = core.distribution.lookup_constructor(name)(*args)
        value = forced_value
        return Clue(value=value, distribution=distribution, forced=True)
    
    return wrapped

def forcing_lookup(statement : Statement, forcing : Mapping[str, Tensor], target : Target):
    """Given a forcing, look up the appropriate callable for a statement."""

    # case 0: the target!
    if statement.function == "target":
        return lift(target)

    # case 1: function is a distribution
    if statement.function in core.distribution.supported_distributions():
        
        # case 1.1: forcing
        if statement.target in forcing.keys():
            return lift_forcing(statement.function, forcing[statement.target])
        
        # case 1.2: no forcing
        else:
            return lift_distribution(statement.function)

    # case 2: function is a builtin
    elif statement.function in core.builtin.supported_builtins():
        return lift(core.builtin.lookup(statement.function))

    # case 3: function can't be found
    else:
        raise KeyError(f"{statement.function} is not a recognized function.")

# SEMANTICS

def semantics_factory(forcing : Mapping[str, Tensor], target : Target) -> Semantics[Clue]:
    """Dynamically construct Spyglass semantics from a forcing."""

    lookup = partial(forcing_lookup, forcing=forcing, target=target)
    return Semantics(Monad(unit, bind), DynamicNamespace(lookup))