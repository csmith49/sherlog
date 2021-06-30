"""Evidence should contain several symbolic constants. One of those symbolic
constants is denoted "the target". Each datum represents an instantiation of
each symbolic constant, including the target."""

from .model import Model
from typing import Dict, Any, Generic, TypeVar, Callable
from ....program import loads, Evidence
from ....inference import Optimizer, Objective
from ..utility import minibatch
from torch import stack, Tensor

T = TypeVar('T')

class Task(Generic[T]):
    """An optimization task."""

    def __init__(self,
        evidence : Evidence,
        target : str,
        input_map : Callable[[T], Dict[str, Any]],
        target_map : Callable[[T], Any]
    ):
        """Build an optimization task.

        Parameters
        ----------
        evidence : Evidence

        target : str

        input_map : Callable[[T], Dict[str, Any]]

        target_map : Callable[[T], Any]
        """
        self.evidence = evidence
        self.target = target
        self.input_map = input_map
        self.target_map = target_map

    def get(self, datum):
        return (self.input_map(datum), self.target_map(datum))

class OptimizationModel(Model):
    """Optimization-based model with uniform evidence and MSE loss."""

    def __init__(self,
        source : str,
        namespace : Dict[str, Any],
        task : Task,
        explanations : int = 1
    ):
        """Build an optimization model.

        Parameters
        ----------
        source : str

        namespace : Dict[str, Any]

        task : Task

        explanations : int (default=1)
        """
        self.program, _ = loads(source, namespace=namespace)
        self.task = task
        # build the explanations from the task evidence - we'll cache these
        self.explanations = self.program.explanations(task.evidence, explanations)

    def datum_loss(self, datum) -> Tensor:
        """Use the task to compute the loss for the given datum.

        Parameters
        ----------
        datum : Any
        """
        results = []
        namespace, target_value = self.task.get(datum)
        
        for explanation in self.explanations:
            # observe the target, modifying explanation in-place
            explanation.observe(self.task.target, target_value)
            loss = explanation.observation_loss(namespace=namespace)
            results.append(loss)

        return stack(results).mean()

    def fit(self, data, *args, epochs : int = 1, batch_size : int = 1, lr : float = 0.001, **kwargs):
        """
        Parameters
        ----------
        data : Iterable[T]

        epochs : int (default=1)

        lr : float (default=0.001)
        """
        # build the optimizer
        optimizer = Optimizer(
            self.program,
            optimizer="adam",
            learning_rate=lr
        )

        # iterate over batches, minimizing datum loss
        for batch in minibatch(data, batch_size, epochs):
            with optimizer as opt:
                # compute loss for each datum in batch
                loss = []
                for datum in batch.data:
                    loss.append(self.datum_loss(datum))
                
                # build the objective and minimize
                objective = Objective(
                    batch.identifier,
                    stack(loss).mean()
                )
                opt.minimize(objective)

    def log_prob(self, datum, *args, **kwargs):
        """Compute log-prob per datum.

        Similar to datum_loss, but uses explanation.log_prob instead of observation_loss.

        Parameters
        ----------
        datum : T
        """
        results = []
        namespace, target_value = self.task.get(datum)
        parameterization = self.program.posterior.paraeterization

        for explanation in self.explanations:
            explanation.observe(self.task.target, target_value)
            log_prob = explanation.log_prob(parameterization, namespace=namespace)
            results.append(log_prob)
        
        return stack(results).mean()