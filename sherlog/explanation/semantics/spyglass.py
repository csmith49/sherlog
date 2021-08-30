from math import dist
from sherlog.pipe.semantics import NDSemantics
from torch.distributions import Distribution
from sherlog.explanation.semantics.core.distribution import supported_distributions
from typing import Callable, List, Mapping, Iterable, Optional, Any
from torch import Tensor, tensor, stack
from functools import partial
from itertools import filterfalse, chain

from ...pipe import DynamicNamespace, Semantics, Pipe, Statement, Value, Literal
from ..observation import Observation

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
        forced : bool = False,
        enumerated : bool = False
    ):
        """Construct a clue."""

        self.value = value
        self._dependencies = list(dependencies) # we'll wrap this
        self.distribution = distribution
        self.forced = forced
        self.enumerated = enumerated

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

    def enumerated_log_prob(self) -> Tensor:
        """Likelihood of producing enumerated value."""

        if self.enumerated:
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
            enumerated.append(dependency.enumerated_log_prob())

        forcing_likelihood = stack(forcing).sum().exp()
        enumerated_likelihood = stack(enumerated).sum().exp()

        tau = stack(mb).sum()
        magic_box = (tau - tau.detach()).exp()

        return self.value * magic_box * forcing_likelihood * enumerated_likelihood

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

    result = callable(*[argument.value for argument in arguments])
    result._dependencies = arguments # never set by callable
    return result

def nd_bind(callable : Callable[..., List[Clue]], arguments : List[Clue]) -> List[Clue]:
    """Evaluate a nondeterministic monadic callable."""

    result = callable(*(argument.value for argument in arguments))
    for clue in result:
        clue._dependencies = arguments # never set by callable
    return result

# SEMANTICS

def lift_deterministic(callable : Callable[..., Tensor]) -> Callable[..., List[Clue]]:
    """Converts a function of type * -> tensor to a function of type * -> clue list."""
    
    def wrapped(*args):
        result = callable(*args)
        return [unit(result)]
    return wrapped

def lift_forced(function_id : str, forced_value : Tensor) -> Callable[..., List[Clue]]:
    """Converts a forced distribution to a function of type * -> clue list."""
    
    def wrapped(*args):
        distribution = core.distribution.lookup_constructor(function_id)(*args)
        clue = Clue(value=forced_value, distribution=distribution, forced=True)
        return [clue]
    return wrapped

def lift_distribution(function_id : str) -> Callable[..., List[Clue]]:
    """Converts a distribution to a function of type * -> clue list."""

    def wrapped(*args):
        distribution = core.distribution.lookup_constructor(function_id)(*args)
        # check if we can enumerate
        try:
            enumerated_values = distribution.enumerate_support().unbind()
            return [Clue(value=value, distribution=distribution, enumerated=True) for value in enumerated_values]
        except NotImplementedError:
            value = distribution.rsample() if distribution.has_rsample else distribution.sample()
            return [Clue(value=value, distribution=distribution)]
    return wrapped

def spyglass_lookup(statement : Statement, forcing : Mapping[str, Tensor], target : Target, locals : Mapping[str, Callable[..., Tensor]]) -> Callable[..., List[Clue]]:
    """Look up the appropriate callable for a statement."""

    # case 0: the target
    if statement.function == "target":
        return lift_deterministic(target)

    # case 1: function is a distribution
    if statement.function in supported_distributions():

        # case 1.1: forced sample
        if statement.target in forcing.keys():
            return lift_forced(statement.function, forcing[statement.target])

        # case 1.2: no forcing, but possibly enumeration
        else:
            return lift_distribution(statement.function)

    # case 2: function is builtin
    if statement.function in core.builtin.supported_builtins():
        return lift_deterministic(core.builtin.lookup(statement.function))

    # case 3: function is local
    if statement.function in locals.keys():
        return lift_deterministic(locals[statement.function])

    # case 4: can't be found
    raise KeyError(f"{statement.functioon} is not a recognized function.")

# namespace

class SpyglassNamespace(DynamicNamespace[List[Clue]]):
    def __init__(self, observation : Observation, target : Target, locals : Mapping[str, Callable[..., Tensor]]):
        """Construct a namespace that forces a given observation."""

        forcing = {k : to_tensor(v.value) for k, v in observation.mapping.items() if isinstance(v, Literal)}
        lookup = partial(spyglass_lookup, forcing=forcing, target=target, locals=locals)

        super().__init__(lookup)

def semantics_factory(observation : Observation, target : Target, locals : Mapping[str, Callable[..., Tensor]]):
    """Builds a set of spyglass semantics that forces a given observation."""

    pipe = Pipe(unit, nd_bind)
    namespace = SpyglassNamespace(observation, target, locals)

    return NDSemantics(pipe, namespace)