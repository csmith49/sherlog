from .objective import Objective
from .bayes import Point, Delta
from ..program import Program
from ..interface import minotaur

from torch import Tensor, tensor, stack
from enum import Enum, auto
from torch.optim import SGD, Adam, Adagrad
from typing import Iterable, Mapping, Optional

import logging

logger = logging.getLogger("sherlog.inference.optimizer")

# managing torch optimization strategies
class Strategy(Enum):
    """Wrapper around Torch optimizers and their parameterizations."""

    SGD = auto(), SGD, {"learning_rate" : "lr", "momentum" : "momentum", "decay" : "weight_decay"}
    ADAM = auto(), Adam, {"learning_rate" : "lr", "decay" : "weight_decay"}
    ADAGRAD = auto(), Adagrad, {"learning_rate" : "lr", "decay" : "weight_decay", "learning_rate_decay" : "lr_decay"}

    @property
    def torch_constructor(self):
        """Get the Torch constructor for the indicated strategy."""

        return self.value[1]

    def filter_kwargs(self, **kwargs):
        """Remove unecessary keyword arguments to satisfy the parameterization of the indicated strategy."""

        result = {}
        mapping = self.value[-1]
        for key, value in kwargs.items():
            try:
                result[mapping[key]] = value
            except KeyError:
                pass
        return result

    def initialize(self, parameters : Iterable[Tensor], **kwargs):
        """Initialize a Torch optimization strategy."""

        return self.torch_constructor(parameters, **self.filter_kwargs(**kwargs))

# managing optimization intent
class Intent(Enum):
    """Enumeration capturing what we want an optimizer to do to an objective: minimize or maximize."""

    MIN = 1
    MAX = -1

    @property
    def sign(self) -> int:
        """When `target * sign` is minimized, `target` will be adjusted according to the intent."""

        return self.value


# optimizer context manager

class Optimizer:
    """Optimizers coordinate program parameter updates from objective-derived gradients.
    
    See also `sherlog.inference.objective`.
    """

    # CONSTRUCTION

    def __init__(self,
        program : Program,
        strategy : Strategy = Strategy.SGD,
        learning_rate : float = 1e-4,
        delta : Optional[Mapping[str, Delta]] = None,
        points : Iterable[Point] = (),
        **kwargs
    ):
        """Construct an optimizer.
        
        Parameters
        ----------
        program : Program

        strategy : Strategy (default=SGD)
            One of `[SGD, Adam]`.

        learning_rate : float (default=1e-4)

        samples : int (default=1)
        """

        self.program = program
        self.strategy = strategy

        self.program_kwargs = kwargs

        self.delta = {} if delta is None else delta

        # construct the store optimizer
        self._optimizer = self.strategy.initialize(self.parameters(points=points), learning_rate=learning_rate)

        # the optimization queue
        self._queue = []

    # PARAMETERS

    def parameters(self, points : Iterable[Point] = ()) -> Iterable[Tensor]:
        for delta in self.delta.values():
            yield from delta.parameters(points)
        yield from self.program.parameters()

    # POINTS

    def lookup_points(self, points : Iterable[Point]) -> Mapping[str, Tensor]:
        result = {}
        
        for point in points:
            delta = self.delta[point.relation]
            result[point.symbol] = delta[point]

        return result

    # HANDLING OBJECTIVES

    def evaluate(self, objective : Objective) -> Tensor:
        """Evaluate an objective to produce a tensor."""
        
        # convert objective parameters and namespace and self.program_kwargs into the arguments for log_prob
        kwargs = {
            "parameters" : self.lookup_points(objective.points),
            **self.program_kwargs
        }

        return self.program.log_prob(objective.evidence, **kwargs)

    # MANAGING THE QUEUE

    def register(self, objective : Objective, intent : Intent):
        """Register an objective to be optimized."""

        # evaluate the objective
        result = self.evaluate(objective)

        # check if it's well-formed (not NaN, not infinite, etc)
        if result.isnan():
            logger.warning(f"Objective produced NaN. [objective={objective}]")
        elif result.isinf():
            logger.warning(f"Objective produced infinite result. [objective={objective}]")
        # and add to the queue if it's good
        else:
            self._queue.append( (result, intent) )

    def maximize(self, *objectives : Objective):
        """Register an objective to be maximized."""

        for objective in objectives:
            self.register(objective, intent=Intent.MAX)

    def minimize(self, *objectives : Objective):
        """Register an objective to be minimized."""
        
        for objective in objectives:
            self.register(objective, intent=Intent.MIN)

    # OPTIMIZATION

    @minotaur("optimizer/optimize")
    def optimize(self) -> Tensor:
        """Update the program parameters to satisfy the collective intent of the provided objectives.
        
        Returns
        -------
        Average computed loss (NaN if no objectives registered).
        """

        # zero the grads in the optimizer
        self._optimizer.zero_grad()
        
        # compute the losses
        losses = [value * intent.sign for value, intent in self._queue]

        if losses:
            loss = stack(losses).sum()
            loss.backward()
            self._optimizer.step()
            self.program.clamp()
            self._queue = []

        else:
            logger.warning("Optimization triggered with an empty optimization queue.")
            loss = tensor(0.0)

        minotaur["batch-loss"] = loss.item()
        return loss

    # CONTEXT MANAGER SEMANTICS
 
    def __enter__(self):
        """Clear the optimization queue and return a reference to `self`."""
        
        return self

    def __exit__(self, *args):
        """Optimize."""

        self.optimize()
