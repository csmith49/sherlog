import torch
from torch.optim import SGD, Adam
from enum import Enum
from ..logs import get
from .objective import Objective

logger = get("optimizer")

# objective intent
Intent = Enum("Intent", "MAXIMIZE MINIMIZE")

class Optimizer:
    """Context manager for registering and optimizing objectives.

    Handles PyTorch optimizers so you don't have to.
    
    See also: `sherlog.inference.Objective`.
    """

    def __init__(self, program, optimizer : str = "sgd", learning_rate : float = 0.1):
        """Constructs an optimization context manager with the indicated Torch optimizer.

        Parameters
        ----------
        problem : Problem

        optimizer : str (default='sgd')
            One of ['sgd', 'adam']

        learning_rate : float (default=0.1)
        """
        self.program = program

        self.optimizer = {
            "sgd" : SGD,
            "adam" : Adam
        }[optimizer](program.parameters(), lr=learning_rate)

        self._maximize, self._minimize = [], []

    def register(self, objective : Objective, intent : Intent = Intent.MAXIMIZE):
        """Registers objectives.

        Parameters
        ----------
        objective : Objective
        intent : Intent
        """
        if intent == Intent.MAXIMIZE:
            self.maximize(objective)
        elif intent == Intent.MINIMIZE:
            self.minimize(objective)

    def maximize(self, *args):
        """Registers objectives to be maximized.

        Parameters
        ----------
        *args : list[Objective]
        """
        for objective in args:
            logger.info(f"Registering {objective} for maximization.")
            if objective.is_nan():
                logger.warning(f"{objective} is NaN.")
            elif objective.is_infinite():
                logger.warning(f"{objective} is infinite.")
            else:
                self._maximize.append(objective)
    
    def minimize(self, *args):
        """Registers objectives to be minimized.

        Parameters
        ----------
        *args : list[Objective]
        """
        for objective in args:
            logger.info(f"Registering {objective} for minimization.")
            if objective.is_nan():
                logger.warning(f"{objective} is NaN.")
            elif objective.is_infinite():
                logger.warning(f"{objective} is infinite.")
            else:
                self._minimize.append(objective)

    def __enter__(self):
        logger.info("Clearing gradients and optimization goals.")
        self.optimizer.zero_grad()
        self._maximize, self._minimize = [], []
        return self

    def __exit__(self, *args):
        cost = torch.tensor(0.0)

        for objective in self._maximize:
            cost -= objective.value
        
        for objective in self._minimize:
            cost += objective.value

        # then update
        logger.info("Propagating gradients.")

        if cost.grad_fn is None:
            logger.warning(f"Cost {cost} has no gradient.")
        else:
            cost.backward()
            self.optimizer.step()
            self.program.clamp()

        for parameter in self.program._parameters:
            logger.info(f"Gradient for {parameter.name}: {parameter.value.grad}")
