from typing import TypeVar, Generic, Iterable, Dict, Any
from abc import ABC, abstractmethod
from torch import Tensor, stack

from ...program import loads
from ...inference import Optimizer, BatchObjective, BatchEmbeddingObjective, Embedding, Objective
from .utility import minibatch

T = TypeVar('T')

class Model(ABC, Generic[T]):
    """Abstract class defining the smallest interface needed to train and evaluate a model."""

    @abstractmethod
    def fit(self, data : Iterable[T], *args, **kwargs):
        pass

    @abstractmethod
    def log_prob(self, datum : T, *args, **kwargs) -> Tensor:
        pass

# build a few kinds of sherlog models
class DirectModel(Model):
    """Model built directly from a Sherlog program."""

    def __init__(self, source : str, namespace : Dict[str, Any]):
        """Build a model from the provided Sherlog program.

        Parameters
        ----------
        source : str
        namespace : Dict[str, Any]
        """
        self._namespace = namespace
        self._program, _ = loads(source, namespace=self._namespace)

    def fit(self, data, *args, epochs=1, lr=0.01, batch_size=1, explanations=1, samples=100, **kwargs):
        """Fit the model to the provided data. Utilizes the default Sherlog machinery.

        Parameters
        ----------
        data : Iterable[Evidence]
        epochs : int (default=1)
        lr : float (default=0.01)
        batch_size : int (default=1)
        explanations : int (default=1)
        samples : int (default=100)
        """
        optimizer = Optimizer(self._program, optimizer="adam", learning_rate=lr)

        for batch in minibatch(data, batch_size, epochs=epochs):
            with optimizer as opt:
                objective = BatchObjective(
                    batch.identifier,
                    self._program,
                    batch.data,
                    log_prob_kwargs = {
                        "explanations" : explanations,
                        "samples" : samples
                    }
                )
                opt.maximize(objective)

        # TODO - make it so we record intermediate results and return them

    def log_prob(self, datum, *args, explanations=1, samples=100, **kwargs):
        return self._program.log_prob(datum, explanations=explanations, samples=samples).item()

class EmbeddingModel(Model):
    """Model built from a Sherlog program, supporting embeddings of arbitrary data."""

    def __init__(self, source : str, namespace : Dict[str, Any], embedding : Embedding):
        """Build a model from a Sherlog program and an embedding.

        Parameters
        ----------
        source : str
        namespace : Dict[str, Any]
        embedding : Embedding
        """
        self._namespace = namespace
        self._program, _ = loads(source, namespace=self._namespace)
        self._embedding = embedding

    def fit(self, data, *args, epochs=1, lr=0.01, batch_size=1, explanations=1, samples=500, **kwargs):
        optimizer = Optimizer(self._program, optimizer="adam", learning_rate=lr)

        for batch in minibatch(data, batch_size, epochs=epochs):
            with optimizer as opt:
                objective = BatchEmbeddingObjective(
                    batch.identifier,
                    self._program,
                    self._embedding,
                    batch.data,
                    log_prob_kwargs = {
                        "explanations" : explanations,
                        "samples" : samples
                    }
                )
                opt.maximize(objective)

    def log_prob(self, sample, *args, explanations=1, samples=500, **kwargs):
        evidence, namespace = self._embedding(sample)
        return self._program.log_prob(evidence, namespace=namespace, explanations=explanations, samples=samples).item()

class SingletonModel(Model):
    def __init__(self, source, namespace, evidence, namespace_generator, explanations=1):
        """
        Parameters
        ----------
        source : str
        namespace : Dict[str, Any]
        evidence : Evidence
        namespace_generator : Callable[[T], Dict[str, Any]]
        explanations : int (default=1)
        """
        # build the usual program
        self._namespace = namespace
        self._program, _ = loads(source, namespace=self._namespace)
        # maintain our single piece of evidence and namespace generator
        self._namespace_generator = namespace_generator
        self._evidence = evidence
        # and pre-compute the explanations we'll use
        self._explanations = list(self._program.explanations(self._evidence, explanations))

    def _datum_log_prob(self, datum, samples):
        # construct our new namespace
        namespace = self._namespace_generator(datum)
        # we need the posterior weights, even if they don't do much here
        parameterization = self._program.posterior.parameterization
        # construct the log probs
        log_probs = [ex.log_prob(parameterization, samples=samples, namespace=namespace) for ex in self._explanations]
        # stack and return
        return stack(log_probs).mean()

    def fit(self, data, *args, epochs=1, lr=0.01, batch_size=1, explanations=1, samples=500, **kwargs):
        optimizer = Optimizer(self._program, optimizer="adam", learning_rate=lr)

        for batch in minibatch(data, batch_size, epochs=epochs):
            with optimizer as opt:
                log_probs = [self._datum_log_prob(datum, samples) for datum in batch.data]
                objective = Objective(
                    batch.identifier,
                    stack(log_probs).mean()
                )
                opt.maximize(objective)

    def log_prob(self, datum, *args, samples=500, **kwargs):
        return self._datum_log_prob(datum, samples).item()