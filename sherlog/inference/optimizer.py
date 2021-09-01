from .objective import Objective
from ..program import Program
from .. import logs

from torch import Tensor, tensor, stack
from enum import Enum, auto
from torch.optim import SGD, Adam

logger = logs.get("optimizer")

# managing torch optimization strategies
Strategy = Enum("Strategy", "SGD ADAM")

_STRATEGY_MAP = {
    Strategy.SGD : SGD,
    Strategy.ADAM : Adam
}

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

    def __init__(self, program : Program, strategy : Strategy = Strategy.SGD, learning_rate : float = 1e-4, **kwargs):
        """Construct an optimizer.
        
        Parameters
        ----------
        program : Program

        strategy : Strategy (default=SGD)
            One of `[SGD, Adam]`.

        learning_rate : float (default=1e-4)
        """

        self.program = program
        self.strategy = strategy

        # group kwargs into strategy and program dicts
        self.strategy_kwargs = {
            "lr" : learning_rate
        }

        self.program_kwargs = {

        }

        # construct the store optimizer
        self._optimizer = _STRATEGY_MAP[strategy](**self.strategy_kwargs)

        # the optimization queue
        self._queue = []

    # HANDLING OBJECTIVES

    def evaluate(self, objective : Objective) -> Tensor:
        """Evaluate an objective to produce a tensor."""
        
        # convert objective parameters and namespace and self.program_kwargs into the arguments for log_prob
        kwargs = {
            "parameters" : objective.parameters,
            **self.program_kwargs
        }

        # if they've provided a conditional, we need two calls to self.program.log_prob
        if objective.conditional:
            numerator = self.program.log_prob(objective.evidence + objective.conditional, **kwargs)
            denominator = self.program.log_prob(objective.conditional, **kwargs)
            # and, because we're in log-space...
            return numerator - denominator
        else:
            return self.program.log_prob(objective.evidence, **kwargs)

    # MANAGING THE QUEUE

    def register(self, objective : Objective, intent : Intent):
        """Register an objective to be optimized."""

        # evaluate the objective
        result = self.evaluate(objective)

        # check if it's well-formed (not NaN, not infinite, etc)
        if result.isnan():
            logger.warning(f"Objective {objective} produced NaN.")
        elif result.isinf():
            logger.warning(f"Objective {objective} produced infinite result.")
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

    def optimize(self) -> Tensor:
        """Update the program parameters to satisfy the collective intent of the provided objectives.
        
        Returns
        -------
        Average computed loss (NaN if no objectives registered).
        """

        logger.info(f"Starting optimization with {self}...")

        # zero the grads in the optimizer
        self._optimizer.zero_grad()
        
        # compute the losses
        losses = [value * intent.sign for value, intent in self._queue]

        if losses:
            loss = stack(losses).sum()
        else:
            logger.warning("Optimization triggered with an empty optimization queue.")
            loss = tensor(0.0)

        # compute gradients and update all the necessary state
        logger.info(f"Updating parameters to minimize loss {loss}...")

        loss.backward()
        self._optimizer.step()
        self.program.clamp()
        
        self._queue = []

        # for debugging, we'll return the average computed loss (NaN if we didn't have any)
        return loss / len(losses)

    # CONTEXT MANAGER SEMANTICS
 
    def __enter__(self):
        """Clear the optimization queue and return a reference to `self`."""
        
        return self

    def __exit__(self, *args):
        """Optimize."""

        self.optimize()
