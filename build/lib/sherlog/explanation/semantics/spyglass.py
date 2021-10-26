from math import dist
from sherlog.pipe.semantics import NDSemantics, ParallelTransform
from torch.distributions import Distribution
from sherlog.explanation.semantics.core.distribution import supported_distributions
from typing import Callable, List, Mapping, Iterable, Optional, Any
from torch import Tensor, tensor, stack
from functools import partial, wraps
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
        enumerated : bool = False
    ):
        """Construct a clue."""

        self.value = value
        self._dependencies = list(dependencies) # we'll wrap this
        self.distribution = distribution
        self.enumerated = enumerated

    def conditional_log_prob(self) -> Tensor:
        """Log-prob of `value` conditioned on the dependencies."""

        if self.distribution:
            return self.distribution.log_prob(self.value)
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
            enumerated.append(dependency.enumerated_log_prob())

        enumerated_likelihood = stack(enumerated).sum().exp()

        tau = stack(mb).sum()
        magic_box = (tau - tau.detach()).exp()

        return self.value * magic_box * enumerated_likelihood

    # magic methods

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"Clue({self.value}, {self.distribution})"

def to_tensor(value : Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    elif isinstance(value, (int, float, tuple, list)):
        return tensor(value)
    elif isinstance(value, bool):
        return tensor(1.0) if value else tensor(0.0)
    else:
        raise TypeError(f"Cannot convert {value} to a tensor.")

def unit(value : Any) -> Clue:
    """Wrap a value in a clue constructor."""

    return Clue(to_tensor(value))

def bind(callable : Callable[..., List[Clue]], arguments : List[Clue]) -> Clue:
    """Evaluate a monadic callable."""

    result = callable(*(argument.value for argument in arguments))
    for clue in result:
        clue._dependencies = arguments # never set by callable
    return result

# SEMANTICS

def lift_deterministic(callable : Callable[..., Tensor]) -> Callable[..., List[Clue]]:
    """Converts a function of type * -> tensor to a function of type * -> clue list."""
    
    @wraps(callable)
    def wrapped(*args):
        result = callable(*args)
        return [unit(result)]
    return wrapped

def lift_distribution(function_id : str) -> Callable[..., List[Clue]]:
    """Converts a distribution to a function of type * -> clue list."""

    def wrapped(*args):
        try:
            distribution = core.distribution.lookup_constructor(function_id)(*args)
        except:
            raise KeyError(f"Cannot construct Spyglass distribution. [function={function_id}, args={args}]")
        # check if we can enumerate
        try:
            enumerated_values = distribution.enumerate_support().unbind()
            return [Clue(value=value, distribution=distribution, enumerated=True) for value in enumerated_values]
        except NotImplementedError:
            value = distribution.rsample() if distribution.has_rsample else distribution.sample()
            return [Clue(value=value, distribution=distribution)]
    return wrapped

def spyglass_lookup(statement : Statement, target : Target, locals : Mapping[str, Callable[..., Tensor]]) -> Callable[..., List[Clue]]:
    """Look up the appropriate callable for a statement."""

    # case 0: the target
    if statement.function == "target":
        return lift_deterministic(target)

    # case 1: function is a distribution
    if statement.function in supported_distributions():
        return lift_distribution(statement.function)

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
    def __init__(self, target : Target, locals : Mapping[str, Callable[..., Tensor]]):
        """Construct a namespace that forces a given observation."""

        lookup = partial(spyglass_lookup, target=target, locals=locals)

        super().__init__(lookup)

def semantics_factory(target : Target, locals : Mapping[str, Callable[..., Tensor]], width : int):
    """Builds a set of spyglass semantics that forces a given observation."""

    pipe = Pipe(unit, bind)
    namespace = SpyglassNamespace(target, locals)

    return ParallelTransform(NDSemantics(pipe, namespace), width)