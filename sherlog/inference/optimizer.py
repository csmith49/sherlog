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
        self._maximize.append(objective)
    
    def minimize(self, objective):
        logger.info(f"Registering {objective} for minimization.")
        self._minimize.append(objective)

    def __enter__(self):
        logger.info("Clearing gradients and optimization goals.")
        self.optimizer.zero_grad()
        self._maximize, self._minimize = [], []
        return self

    def __exit__(self, *args):
        # construct storch costs
        for objective in self._maximize:
            storch.add_cost(-1 * objective.value, objective.name)
        
        for objective in self._minimize:
            storch.add_cost(objective.value, objective.name)

        # compute gradients
        if self._maximize or self._minimize:
            storch.backward()
        
        # then update
        logger.info("Propagating gradients.")
        self.optimizer.step()
        self.problem.clamp_parameters()