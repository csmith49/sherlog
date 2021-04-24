import torch
from torch.optim import SGD, Adam
from ..logs import get

logger = get("optimizer")

class Optimizer:
    def __init__(self, program, optimizer : str = "sgd", learning_rate : float = 0.1):
        """Context manager for optimizing registered objectives.

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
            self.program.clamp_parameters()

        for p, v in self.program.parameter_map.items():
            logger.info(f"Gradient for {p}: {v.grad}.")