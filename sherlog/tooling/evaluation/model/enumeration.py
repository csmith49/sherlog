"""Evidence should contain several symbolic constants. One of those symbolic
constants is denoted "the target". Each datum represents an instantiation of
each symbolic constant, including the target."""

from .model import Model, Task
from typing import Dict, Any, Generic, TypeVar, Callable, List
from ....program import loads, Evidence
from ....inference import Optimizer, Objective
from ....engine import Store
from ..utility import minibatch
from torch import stack, Tensor

T = TypeVar('T')

class EnumerationModel(Model[T]):
    """Enumeration-based model with uniform evidence and indicator loss."""

    def __init__(self, source : str, task : Task[T], bindings : Dict[str, Any], explanations : int = 1):
        """Build an enumeration model.
        
        Parameters
        ----------
        source : str
        
        task : Task[T]
        
        bindings : Dict[str, Any]
        
        explanations : int (default=1)
        """
        self.program, _ = loads(source, namespace=bindings)
        self.task = task

        # build explanations from task evidence - we'll cache 'em and reuse whenever possible
        self.explanations = list(self.program.explanations(task.evidence, quantity=explanations))

    def store(self, datum : T) -> Store:
        """Construct an execution store for the given datum.
        
        Parameters
        ----------
        datum : T
        
        Returns
        -------
        Store
        """
        return self.program.store(**self.task.inject(datum))

    def log_prob(self, datum : T, *args, samples : int = 100, **kwargs) -> Tensor:
        """Log-likelihood of the given datum.
        
        Parameters
        ----------
        datum : T
        
        *args
            Unused.
        
        samples : int (default=100)
            Number of samples used to stochastically estimate the log-likelihood.

        **kwargs
            Unused.
            
        Returns
        -------
        Tensor
        """
        results = []
        for _ in range(samples):
            store = self.store(datum)
            prob = stack([ex.log_prob(store).exp() for ex in self.explanations]).mean()
            results.append(prob)
        return stack(results).mean().log()
        # return stack([ex.log_prob(store) for ex in self.explanations]).mean()

    def fit(self, data : List[T], *args, epochs : int = 1, batch_size : int = 20, lr : float = 0.01, **kwargs):
        """Fit the model to the given data.
        
        Parameters
        ----------
        data : List[T]
        
        *args
            Unused.
            
        epochs : int (default=1)
        
        batch_size : int (default=1)
        
        lr : float (default=0.001)
        
        **kwargs
            Unused.
        """
        optimizer = Optimizer(self.program, optimizer="adam", learning_rate=lr)

        for batch in minibatch(data, batch_size, epochs):
            with optimizer as opt:
                lp = stack([self.log_prob(datum) for datum in batch.data]).mean()
                objective = Objective(batch.identifier, lp)
                opt.maximize(objective)