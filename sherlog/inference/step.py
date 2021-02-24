import storch
from ..logs import get

logger = get("inference.step")

class StepContextManager:
    def __init__(self, optimizer, problem):
        self.optimizer, self.problem = optimizer, problem

    def __enter__(self):
        logger.info("Resetting gradients...")
        self.optimizer.zero_grad()

    def __exit__(self, *args):
        logger.info("Computing gradients...")
        storch.backward()
        self.optimizer.step()
        self.problem.clamp_parameters()
        logger.info("Gradients updated.")

def step(optimizer, problem):
    return StepContextManager(optimizer, problem)