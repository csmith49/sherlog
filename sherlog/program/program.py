from ..logs import get

logger = get("program")

from .evidence import Evidence
from .parameter import Parameter
from .posterior import LinearPosterior

from ..explanation import Explanation
from ..interface import query

from typing import Optional, Iterable, Mapping, Any
from itertools import islice
from torch import Tensor, tensor, stack, no_grad
from random import random

class Program:
    """Programs coordinate the generation of explanations."""

    def __init__(self, source, parameters, locals : Optional[Mapping[str, Any]] = None):
        self._source = source
        self._parameters = list(parameters)
        self._locals = locals if locals else {}

        self.posterior = LinearPosterior()

    @classmethod
    def of_json(cls, json, locals : Optional[Mapping[str, Any]] = None) -> 'Program':
        """Build a program from a JSON-like object."""
        parameters = [Parameter.of_json(parameter) for parameter in json["parameters"]]
        rules = json["rules"]
        return cls(rules, parameters)

    def explanations(self, evidence : Evidence, quantity : int = 1, attempts : int = 100, width : Optional[int] = None) -> Iterable[Explanation]:
        """Sample explanations for the provided evidence."""

        logger.info(f"Sampling explanations for evidence {evidence}...")

        # build kwargs for queries
        kwargs = {}
        kwargs["width"] = width
        kwargs["contexts"] = []
        kwargs["parameterization"] = self.posterior.parameterization()

        # build generator
        def gen():
            for attempt in range(attempts):
                logger.info(f"Starting explanation generation attempt {attempt}...")

                try:
                    for json in query(self._source, evidence.json, **kwargs):
                        yield Explanation.load(json)
                except TimeoutError:
                    logger.warning(f"Explanation generation attempt {attempt} timed out. Restarting...")

        # get at most quantity explanations
        yield from islice(gen(), quantity)

    def log_prob(self, evidence : Evidence, explanations : int = 1, attempts = 100, width : Optional[int] = None, locals : Optional[Mapping[str, Any]] = None) -> Tensor:
        """Compute the marginal log-likelihood of the provided evidence."""

        logger.info(f"Evaluating log-prob for {evidence}...")

        # build -> sample -> evaluate
        explanations = self.explanations(evidence, quantity=explanations, attempts=attempts, width=width)
        log_probs = [explanation.log_prob() - self._posterior.log_prob(explanation) for explanation in explanations]

        # if no explanations, default
        if log_probs:
            result = stack(log_probs).mean()
            logger.info(f"Log-prob for {evidence}: {result:f}.")
        else:
            result = tensor(0.0)
            logger.info("No explanations generated. Defaulting to log-prob of 0.0.")

        return result

    def parameters(self, locals : Optional[Mapping[str, Any]] = None) -> Iterable[Tensor]:
        """Yields all tuneable parameters in the program and optional local namespace."""

        # handle params
        for parameter in self._parameters:
            yield parameter.value

        # handle internal namespace
        for obj in self._locals.values():
            if hasattr(obj, "parameters"):
                yield from obj.parameters()

        # handle external namespace
        if locals:
            for obj in locals.values():
                if hasattr(obj, "parameters"):
                    yield from obj.parameters()

        # handle posterior
        yield from self._posterior.parameters()

    def clamp(self):
        """Update program parameters in-place to satisfy their domain constraints."""

        logger.info(f"Clamping parameters for {self}...")

        with no_grad():
            for parameter in self._parameters:
                parameter.clamp()

    def sample_explanation(self, evidence : Evidence, burn_in : int = 100, locals : Optional[Mapping[str, Any]] = None, **kwargs) -> Explanation:
        """Samples an explanation from the posterior."""

        logger.info(f"Sampling explanation for {evidence} with {burn_in} burn-in steps.")

        asmple, sample_likelihood = None, 0.00001

        for step in range(burn_in):
            # sample a new explanation and compute likelihood
            explanation = next(self.explanations(evidence, quantity=1, **kwargs))
            explanation_likelihood = explanation.log_prob().exp()

            # accept / reject
            ratio = explanation_likelihood / sample_likelihood
            if random.random() <= ratio:
                logger.info(f"Step {step}: sample accepted with likelihood ratio {ratio}.")
                sample, sample_likelihood = explanation, explanation_likelihood

        if sample is None:
            logger.warning(f"No sample accepted after {burn_in} burn-in steps.")

        return sample