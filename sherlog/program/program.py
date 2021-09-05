from ..interface.logs import get

logger = get("program")

from .evidence import Evidence
from .parameter import Parameter
from .posterior import UniformPosterior

from ..explanation import Explanation
from ..interface import query

from typing import Optional, Iterable, Mapping, Any, Callable
from itertools import islice
from torch import Tensor, tensor, stack, no_grad
from random import random

class Program:
    """Programs coordinate the generation of explanations."""

    def __init__(self, source, parameters, locals : Mapping[str, Callable[..., Tensor]]):
        self._source = source
        self._parameters = list(parameters)
        self._locals = locals

        self.posterior = UniformPosterior()

    @classmethod
    def of_json(cls, json, locals : Optional[Mapping[str, Any]] = None) -> 'Program':
        """Build a program from a JSON-like object."""
        parameters = [Parameter.of_json(parameter) for parameter in json["parameters"]]
        rules = json["rules"]
        return cls(rules, parameters, locals=locals if locals else {})

    # EXPLANATION EVALUATION
    def store(self, **kwargs : Tensor) -> Mapping[str, Tensor]:
        return {**kwargs, **{parameter.name : parameter.value for parameter in self._parameters}}

    def explanations(self, evidence : Evidence, quantity : int = 1, attempts : int = 100, width : Optional[int] = None) -> Iterable[Explanation]:
        """Sample explanations for the provided evidence."""

        logger.info(f"Sampling explanations for evidence {evidence}...")

        # build generator
        def gen():
            for attempt in range(attempts):
                logger.info(f"Starting explanation generation attempt {attempt}...")

                try:
                    for json in query(self._source, evidence.to_json(), self.posterior.to_json(), width=width):
                        yield Explanation.of_json(json, locals=self._locals)
                except TimeoutError:
                    logger.warning(f"Explanation generation attempt {attempt} timed out. Restarting...")

        # get at most quantity explanations
        yield from islice(gen(), quantity)

    def log_prob(self,
        evidence : Evidence,
        explanations : int = 1,
        attempts = 100,
        width : Optional[int] = None,
        parameters : Optional[Mapping[str, Tensor]] = None,
    ) -> Tensor:
        """Compute the marginal log-likelihood of the provided evidence."""

        logger.info(f"Evaluating log-prob for {evidence}...")

        # build -> sample -> evaluate
        store = self.store(**(parameters if parameters else {}))
        explanations = self.explanations(evidence, quantity=explanations, attempts=attempts, width=width)
        # log_probs = [explanation.log_prob(store) - self.posterior.log_prob(explanation) for explanation in explanations]
        log_probs = [explanation.log_prob(store) for explanation in explanations]

        # if no explanations, default
        if log_probs:
            result = stack(log_probs).mean()
            logger.info(f"Log-prob for {evidence}: {result:f}.")
        else:
            result = tensor(0.0)
            logger.info("No explanations generated. Defaulting to log-prob of 0.0.")

        return result

    def conditional_log_prob(self,
        evidence : Evidence, condition : Evidence,
        explanations : int = 1,
        attempts = 100,
        width : Optional[int] = None,
        parameters : Optional[Mapping[str, Tensor]] = None
    ) -> Tensor:
        """Compute the log-likelihood of the provided evidenced conditioned on another piece of evidence."""

        numerator = self.log_prob(evidence.join(condition),
            explanations=explanations,
            attempts=attempts,
            width=width,
            parameters=parameters
        )

        denominator = self.log_prob(condition,
            explanations=explanations,
            attempts=attempts,
            witdth=width,
            parameters=parameters
        )

        return numerator - denominator

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
        yield from self.posterior.parameters()

    def clamp(self):
        """Update program parameters in-place to satisfy their domain constraints."""

        logger.info(f"Clamping parameters for {self}...")

        with no_grad():
            for parameter in self._parameters:
                parameter.clamp()

    def sample_posterior(self, evidence : Evidence, burn_in : int = 100) -> Iterable[Explanation]:
        """Sample explanation from the posterior.
        
        Note, this actually samples from the prior right now.
        """

        logger.info(f"Sampling explanations from the posterior for {evidence} with {burn_in} burn-in steps.")

        sample, sample_likelihood = None, 0.0001
        for step in range(burn_in):
            # sample a new explanation and compute likelihood
            explanation = next(self.explanations(evidence, quantity=1))
            store = self.store()
            explanation_likelihood = explanation.log_prob(store).exp()

            # accept / reject
            ratio = explanation_likelihood / sample_likelihood
            if random() <= ratio:
                logger.info(f"Step {step}: sample accepted with likelihood ratio {ratio}.")
                sample, sample_likelihood = explanation, explanation_likelihood

        if sample is None:
            logger.warning(f"No sample accepted after {burn_in} burn-in steps.")

        return sample