from .evidence import Evidence
from .parameter import Parameter
from .posterior import Posterior

from ..explanation import Explanation
from ..interface import query, minotaur

from typing import Optional, Iterable, Mapping, Any, Callable
from torch import Tensor, no_grad
from random import random

import logging

logger = logging.getLogger("sherlog.program")

class Program:
    """Programs coordinate the generation of explanations."""

    def __init__(self, rules, parameters, posterior, locals : Mapping[str, Callable[..., Tensor]]):
        self._rules = rules
        self._parameters = list(parameters)
        self._posterior = posterior
        
        self._locals = locals

        # cache for explanation queries
        self._cache = {}

    # IO

    @classmethod
    def of_json(cls, json, locals : Optional[Mapping[str, Any]] = None) -> 'Program':
        """Build a program from a JSON-like object."""

        rules = json["rules"]
        parameters = [Parameter.of_json(parameter) for parameter in json["parameters"]]
        posterior = Posterior.of_json(json["posterior"])

        return cls(rules, parameters, posterior, locals=locals if locals else {})

    def to_json(self):
        """Return a JSON representation of the program."""
        # TODO: fix parameter representation; they're not used for queries, but they *might* be later

        return {
            "type" : "program",
            "rules" : self._rules,
            "parameters" : [],
            "posterior" : self._posterior.to_json()
        }

    # EXPLANATION EVALUATION

    def store(self, **kwargs : Tensor) -> Mapping[str, Tensor]:
        """Evaluation store for explanations generated from the program."""

        return {**kwargs, **{parameter.name : parameter.value for parameter in self._parameters}}

    @minotaur("program/explanation", kwargs=("cache"))
    def explanation(self, evidence : Evidence, cache : bool = False) -> Explanation:
        """Sample explanations for the provided evidence."""
        
        # check if the evidence is cached
        if cache and evidence in self._cache.keys():
            explanation = self._cache[evidence]
        else:
            json = query(self, evidence)
            explanation = Explanation.of_json(json, locals=self._locals)
             
        # cache the explanation if requested
        if cache:
            self._cache[evidence] = explanation

        return explanation

    @minotaur("program/log-prob", kwargs=("attempts", "samples", "cache"))
    def log_prob(self, evidence : Evidence, samples : int = 1, parameters : Optional[Mapping[str, Tensor]] = None, force : bool = False, cache : bool = False) -> Tensor:
        """Compute the marginal log-likelihood of the provided evidence."""

        # build -> sample -> evaluate
        store = self.store(**(parameters if parameters else {}))

        explanation = self.explanation(evidence, cache=cache)
        result = explanation.log_prob(store, samples=samples, force=force)

        minotaur["result"] = result.item()
        return result

    @minotaur("program/conditional-log-prob", kwargs=("attempts", "cache"))
    def conditional_log_prob(self,
        evidence : Evidence, condition : Evidence,
        attempts = 100,
        parameters : Optional[Mapping[str, Tensor]] = None,
        cache : bool = False
    ) -> Tensor:
        """Compute the log-likelihood of the provided evidenced conditioned on another piece of evidence."""

        numerator = self.log_prob(evidence.join(condition),
            attempts=attempts,
            parameters=parameters,
            cache=cache
        )

        denominator = self.log_prob(condition,
            attempts=attempts,
            parameters=parameters,
            cache=cache
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
        yield from self._posterior.parameters()

    def parameter(self, name : str) -> Tensor:
        """Look up a parameter by name.
        
        Will not consider parameters of embedded function approximators."""

        for parameter in self._parameters:
            if parameter.name == name:
                return parameter.value

        return None

    def clamp(self):
        """Update program parameters in-place to satisfy their domain constraints."""

        with no_grad():
            for parameter in self._parameters:
                parameter.clamp()

    @minotaur("program/sample-posterior", kwargs=("burn in"))
    def sample_posterior(self, evidence : Evidence, burn_in : int = 100) -> Iterable[Explanation]:
        """Sample explanation from the posterior."""

        # TODO - this actually samples from the prior right now; fix this discrepancy

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