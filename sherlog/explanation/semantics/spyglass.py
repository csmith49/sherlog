from torch.distributions import Distribution
from sherlog.explanation.semantics.core.distribution import supported_distributions
from typing import Callable, List, Mapping, Iterable, Optional, Any
from torch import Tensor, tensor, stack
from functools import partial, wraps
from itertools import filterfalse, chain

from ...pipe import DynamicNamespace, Semantics, Pipe, Statement, Literal
from ..observation import Observation

from .core.target import Target
from . import core

# UTILITY

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

def to_tensor(value : Any) -> Tensor:
    """Utility for converting objects to tensors in a reasonable way."""
    
    if isinstance(value, Tensor):
        return value
    elif isinstance(value, (int, float, tuple, list)):
        return tensor(value)
    elif isinstance(value, bool):
        return tensor(1.0) if value else tensor(0.0)
    else:
        raise TypeError(f"Cannot convert {value} to a tensor.")

# MONADIC SEMANTICS

class Clue:
    """Monadic constructor for the Spyglass algorithm."""

    def __init__(self,
        value : Tensor,
        dependencies : Iterable['Clue'] = (),
        distribution : Optional[Distribution] = None,
        forced : bool = False,
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
        mb, forcing, enumerated = [tensor(0.0)], [tensor(0.0)], [tensor(0.0)]

        # construct magic box and forcing scaling values in one pass
        for dependency in self.dependencies:
            mb.append(dependency.conditional_log_prob())
            forcing.append(dependency.forcing_log_prob())

        forcing_likelihood = stack(forcing).sum().exp()

        tau = stack(mb).sum()
        magic_box = (tau - tau.detach()).exp()

        return self.value * magic_box * forcing_likelihood

    # magic methods

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"Clue({self.value}, {self.distribution})"

# MONADIC EVALUATION

def unit(value : Any) -> Clue:
    """Wrap a value in a clue constructor."""

    return Clue(to_tensor(value))

def bind(callable : Callable[..., Clue], arguments : List[Clue]) -> Clue:
    """Evaluate a monadic callable on a list of monadic arguments."""

    result = callable(*[argument.value for argument in arguments])
    result._dependencies = arguments # never set by callable
    return result

# SEMANTICS

def lift_deterministic(callable : Callable[..., Tensor]) -> Callable[..., Clue]:
    """Converts a function of type * -> tensor to a function of type * -> Clue."""

    @wraps(callable)
    def wrapped(*arguments):
        return unit(callable(*arguments))
    return wrapped

def lift_distribution(distribution_constructor, forcing : Optional[Tensor] = None):
    """Converts a distribution constructor to a function of type * -> Clue."""

    @wraps(distribution_constructor)
    def wrapped(*arguments):
        distribution = distribution_constructor(*arguments)

        # short-circuit sampling if there's a forcing given
        if forcing is not None:
            return Clue(value=forcing, distribution=distribution, forced=True)
        
        # otherwise just return a sample (reparameterized, if possible)
        else:
            sample = distribution.rsample() if distribution.has_rsample else distribution.sample()
            return Clue(value=sample, distribution=distribution)

    return wrapped

def spyglass_lookup(statement : Statement, forcing : Observation, target : Target, locals : Mapping[str, Callable[..., Tensor]]) -> Callable[..., List[Clue]]:
    """Look up the appropriate callable for a statement."""

    # case 0: the target
    if statement.function == "target":
        return lift_deterministic(target)

    # case 1: function is a distribution
    if statement.function in supported_distributions():
        distribution_constructor = core.distribution.lookup_constructor(statement.function)
        forced_value = forcing[statement.target] if statement.target in forcing.keys() else None
        return lift_distribution(distribution_constructor, forcing=forced_value)

    # case 2: function is builtin
    if statement.function in core.builtin.supported_builtins():
        return lift_deterministic(core.builtin.lookup(statement.function))

    # case 3: function is local
    if statement.function in locals.keys():
        return lift_deterministic(locals[statement.function])

    # case 4: can't be found
    raise KeyError(f"{statement.function} is not a recognized function.")

# namespace

class SpyglassNamespace(DynamicNamespace[List[Clue]]):
    def __init__(self, forcing : Observation, target : Target, locals : Mapping[str, Callable[..., Tensor]]):
        """Construct a namespace that forces a given observation."""

        lookup = partial(spyglass_lookup, forcing=forcing, target=target, locals=locals)

        super().__init__(lookup)

def semantics_factory(observation : Observation, target : Target, locals : Mapping[str, Callable[..., Tensor]], width : int):
    """Builds a set of spyglass semantics that forces a given observation."""

    pipe = Pipe(unit, bind)
    namespace = SpyglassNamespace(observation, target, locals)

    return Semantics(pipe, namespace)
