import storch
import torch
from ..logs import get

logger = get("inference.optimizer")

class Optimizer:
    def __init__(self, problem, optimizer):
        self.problem = problem
        self.optimizer = optimizer

        self._maximize, self._minimize = [], []

    def maximize(self, objective):
        logger.info(f"Registering {objective} for maximization.")
        if objective.is_nan():
            logger.warning(f"{objective} is NaN.")
        self._maximize.append(objective)
    
    def minimize(self, objective):
        logger.info(f"Registering {objective} for minimization.")
        if objective.is_nan():
            logger.warning(f"{objective} is NaN.")
        self._minimize.append(objective)

    def __enter__(self):
        logger.info("Clearing gradients and optimization goals.")
        self.optimizer.zero_grad()
        self._maximize, self._minimize = [], []
        return self

    def __exit__(self, *args):
        cost = torch.tensor(0.0)

        # construct storch costs
        for objective in self._maximize:
            cost -= objective.value

        # compute gradients
        if self._maximize or self._minimize:
            cost.backward()
        else:
            logger.warning("No objectives registered.")
        
        # then update
        logger.info("Propagating gradients.")
        self.optimizer.step()
        self.problem.clamp_parameters()