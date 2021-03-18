import torch
from ..logs import get

logger = get("optimizer")

class Optimizer:
    def __init__(self, problem, optimizer):
        self.problem = problem
        self.optimizer = optimizer

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
        self.optimizer.step()
        self.problem.clamp_parameters()

        for p, v in self.problem.parameter_map.items():
            logger.info(f"Gradient for {p}: {v.grad}.")