from torch.distributions import Distribution
from sherlog.explanation.semantics.core.distribution import supported_distributions
from typing import Callable, List, Mapping, Iterable, Optional, Any
from torch import Tensor, tensor, stack
from functools import partial
from itertools import filterfalse, chain

from ...pipe import DynamicNamespace, Semantics, Pipe, Statement, Value, Literal

from .core.target import Target
from . import core

# utility for ensuring uniqueness in enumeration
def unique(iterable, key=None):
    """List unique elements by `key`, if provided."""

    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

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

def to_tensor(value : Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    elif isinstance(value, (int, float)):
        return tensor(value)
    elif isinstance(value, bool):
        return tensor(1.0) if value else tensor(0.0)
    else:
        raise TypeError(f"Cannot convert {value} to a tensor.")

def unit(value : Any) -> Clue:
    """Wrap a value in a clue constructor."""

    return Clue(to_tensor(value))

def bind(callable : Callable[..., Clue], arguments : List[Clue]) -> Clue:
    """Evaluate a monadic callable on a list of monadic arguments."""

    result = callable(*(argument.value for argument in arguments))
    result._dependencies = arguments # never set by callable
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
        value = distribution.rsample() if distribution.has_rsample else distribution.sample()
        return Clue(value=value, distribution=distribution)
    
    return wrapped

def lift_forcing(name : str, forced_value : Tensor) -> Callable[..., Clue]:
    """Lift a distribution identifier and forced value to a monadic function."""

    def wrapped(*args):
        distribution = core.distribution.lookup_constructor(name)(*args)
        value = forced_value
        return Clue(value=value, distribution=distribution, forced=True)
    
    return wrapped

def forcing_lookup(statement : Statement, forcing : Mapping[str, Tensor], target : Target, locals : Mapping[str, Callable[..., Tensor]]):
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

    # case 3: check the local functions
    elif statement.function in locals.keys():
        return lift(locals[statement.function])

    # case 4: function can't be found
    else:
        raise KeyError(f"{statement.function} is not a recognized function.")

# SEMANTICS

def semantics_factory(forcing : Mapping[str, Value], target : Target, locals : Optional[Mapping[str, Callable[..., Tensor]]] = None) -> Semantics[Clue]:
    """Dynamically construct Spyglass semantics from a forcing."""

    forced_tensors = {k : to_tensor(v.value) for k, v in forcing.items() if isinstance(v, Literal)}

    lookup = partial(forcing_lookup, forcing=forced_tensors, target=target, locals=locals if locals else {})
    return Semantics(Pipe(unit, bind), DynamicNamespace(lookup))