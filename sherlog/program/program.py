from .evidence import Evidence
from .parameter import Parameter
from .posterior import Posterior

from ..explanation import Explanation
from ..interface import query
from ..interface.instrumentation import minotaur

from typing import Optional, Iterable, Mapping, Any, Callable
from itertools import islice
from torch import Tensor, tensor, stack, no_grad
from random import random

import logging

logger = logging.getLogger("sherlog.program")

class Program:
    """Programs coordinate the generation of explanations."""

    def __init__(self, rules, parameters, posterior, locals : Mapping[str, Callable[..., Tensor]]):
        self._rules = rules
        self._parameters = list(parameters)
        self._locals = locals

        self.posterior = posterior

    @classmethod
    def of_json(cls, json, locals : Optional[Mapping[str, Any]] = None) -> 'Program':
        """Build a program from a JSON-like object."""

        rules = json["rules"]
        parameters = [Parameter.of_json(parameter) for parameter in json["parameters"]]
        posterior = Posterior.of_json(json["posterior"])

        return cls(rules, parameters, posterior, locals=locals if locals else {})

    def dump(self):
        return {
            "type" : "program",
            "rules" : self._rules,
            "parameters" : [], #TODO: fix this
            "posterior" : self.posterior.dump()
        }

    # EXPLANATION EVALUATION
    def store(self, **kwargs : Tensor) -> Mapping[str, Tensor]:
        return {**kwargs, **{parameter.name : parameter.value for parameter in self._parameters}}

    def explanation(self, evidence : Evidence, attempts : int = 100) -> Explanation:
        """Sample explanations for the provided evidence."""

        for _ in range(attempts):
            try:
                json = query(self, evidence)
                return Explanation.of_json(json, locals=self._locals)
            except TimeoutError:
                pass
        raise TimeoutError(f"Explanation sampling timed-out. [evidence={evidence}, attempts={attempts}]")

    @minotaur("log-prob", kwargs=("explanations"))
    def log_prob(self,
        evidence : Evidence,
        attempts = 100,
        parameters : Optional[Mapping[str, Tensor]] = None,
    ) -> Tensor:
        """Compute the marginal log-likelihood of the provided evidence."""

        # build -> sample -> evaluate
        store = self.store(**(parameters if parameters else {}))

        explanation = self.explanation(evidence, attempts=attempts)
        result = explanation.log_prob(store)

        minotaur["result"] = result.item()
        print(result)
        return result

    @minotaur("conditional log-prob", kwargs=("explanations"))
    def conditional_log_prob(self,
        evidence : Evidence, condition : Evidence,
        attempts = 100,
        parameters : Optional[Mapping[str, Tensor]] = None
    ) -> Tensor:
        """Compute the log-likelihood of the provided evidenced conditioned on another piece of evidence."""

        numerator = self.log_prob(evidence.join(condition),
            attempts=attempts,
            parameters=parameters
        )

        denominator = self.log_prob(condition,
            attempts=attempts,
            parameters=parameters
        )

        minotaur["numerator"] = numerator.item()
        minotaur["denominator"] = denominator.item()
        minotaur["result"] = (numerator - denominator).item()
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

        with no_grad():
            for parameter in self._parameters:
                parameter.clamp()

    @minotaur("sample", kwargs=("burn in"))
    def sample_posterior(self, evidence : Evidence, burn_in : int = 100) -> Iterable[Explanation]:
        """Sample explanation from the posterior.
        
        NOTE: this actually samples from the prior right now.
        TODO: fix this discrepancy.
        """

        sample, sample_likelihood = None, 0.0001
        for step in range(burn_in):
            with minotaur("step"):
                # sample a new explanation and compute likelihood
                explanation = next(self.explanations(evidence, quantity=1))
                store = self.store()
                explanation_likelihood = explanation.log_prob(store).exp()

                minotaur["likelihood"] = explanation_likelihood.item()

                # accept / reject
                ratio = explanation_likelihood / sample_likelihood
                if random() <= ratio:
                    logger.info(f"Step {step}: sample accepted with likelihood ratio {ratio}.")
                    sample, sample_likelihood = explanation, explanation_likelihood
                    minotaur["result"] = True
                else:
                    minotaur["result"] = False

        if sample is None:
            logger.warning(f"No sample accepted after {burn_in} burn-in steps.")

        return sample