from .parameter import Parameter
from .evidence import Evidence
from .posterior import Posterior
from .. import interface
from ..explanation import Explanation
from ..logs import get

from itertools import islice, chain, repeat
from typing import Iterable, Optional

import torch
import pickle
import random

logger = get("program")

class Program:
    def __init__(self, parameters, program_source, namespace, contexts=None):
        """Object representing a Sherlog problem file.

        Parameters
        ----------
        parameters : Iterable[Parameter]

        namespaces : Iterable[str]

        posterior : Optional[List[str]]

        program_source : JSON
        """
        # convert params to name-param map
        self._parameters = {p.name : p for p in parameters}
        
        self._namespace = namespace
        
        # just record the rest
        self.program_source = program_source

        # and set the posterior
        if contexts:
            self.posterior = Posterior(contexts=contexts)
        else:
            self.posterior = Posterior()

    @classmethod
    def of_json(cls, json, namespace=None):
        """Build a program from a JSON-like object.

        Parameters
        ----------
        json : JSON
        
        Returns
        -------
        Program
        """
        parameters = [Parameter.of_json(p) for p in json["parameters"]]
        # namespaces = json["namespaces"]
        program_source = {
            "rules" : json["rules"],
            "parameters" : json["parameters"],
            "ontology" : json["ontology"],
            "evidence" : [] # we split the evidence out into a different iterable, but the server expects this
        }
        if namespace:
            return cls(parameters, program_source, namespace)
        else:
            return cls(parameters, program_source, {})

    def explanations(self, evidence : Evidence, quantity : int, attempts : int = 100, width : Optional[int] = None, namespace = None, posterior = None):
        """Samples explanations for the provided evidence.

        Parameters
        ----------
        evidence : Evidence
        quantity : int
        attempts : int (default=100)
        width : Optional[int]
        namespace : Optional[Mapping[str, Any]]
        posterior : Optional[Posterior]

        Returns
        -------
        Iterable[Explanation]
        """
        logger.info("Sampling explanations for evidence %s...", evidence)

        # build the external evaluation context w/ namespaces
        if namespace:
            external = (self.parameter_map, self._namespace, namespace)
        else:
            external = (self.parameter_map, self._namespace)

        # interface kwargs
        kwargs = {}
        kwargs["width"] = width
        if posterior:
            kwargs["contexts"] = list(posterior.contexts)
            kwargs["parameterization"] = list(posterior.weights)

        # build the explanation generator
        def gen():
            for attempt in range(attempts):
                logger.info("Starting explanation generation attempt %i...", attempt)
                try:
                    for json in interface.query(self.program_source, evidence.json, **kwargs):
                        logger.info("Explanation found.")
                        yield Explanation.of_json(json, external=external)
                except TimeoutError:
                    logger.warning("Explanation generation timed out. Restarting...")
        
        yield from islice(gen(), quantity)

    def log_prob(self, evidence, explanations=1, samples=100, attempts=100, width=100, namespace=None):
        """Compute the marginal log-likelihood of the provided evidence.

        Parameters
        ----------
        evidence : Evidence
        explanations : int (default=1)
        samples : int (default=100)
        attempts : int (default=100)
        width : int (default=100)
        namepsace : Optional[Namespace]

        Returns
        -------
        Tensor
        """
        explanations = self.explanations(evidence, quantity=explanations, width=width, attempts=attempts, namespace=namespace)
        log_probs = [ex.log_prob(self.posterior.parameterization, samples=samples) for ex in explanations]
        # if we didn't find any explanations, default
        if log_probs:
            return torch.mean(torch.stack(log_probs))
        else:
            return torch.ones(samples)

    def sample_explanation(self, evidence : Evidence, burn_in : int = 100, samples : int = 100, **kwargs):
        """Sample an explanation from the posterior.

        Parameters
        ----------
        evidence : Evidence
        burn_in : int (default=100)
        samples : int (default=100)

        Returns
        -------
        Explanation
        """

        logger.info(f"Sampling explanation for {evidence} with {burn_in} burn-in steps.")
        sample, sample_likelihood = None, 0.00001

        for step in range(burn_in):
            # sample a new explanation and compute likelihood
            explanation = next(self.explanations(evidence, quantity=1, **kwargs))
            # compute the likelihood
            explanation_likelihood = explanation.miser(samples=samples).mean().item()
            # accept/reject
            ratio = explanation_likelihood / sample_likelihood
            if random.random() <= ratio:
                logger.info(f"Step {step}: sample accepted with likelihood ratio {ratio}.")
                sample, sample_likelihood = explanation, explanation_likelihood

        if sample is None:
            logger.warning(f"No sample accepted after {burn_in} burn-in steps.")
        return sample

    # TODO - save and load posterior parameters with this contraption

    def save_parameters(self, filepath):
        """Write all parameter values in scope to a file.

        Parameters
        ----------
        filepath : str
        """
        output = { "parameters" : {}, "models" : {} }
        for p, parameter in self._parameters.items():
            output["parameters"][p] = parameter.value
        for name, obj in self._namespace.items():
            if hasattr(obj, "state_dict"):
                output["models"][name] = obj.state_dict()
        with open(filepath, "wb") as f:
            pickle.dump(output, f)

    def load_parameters(self, filepath):
        """Update (in-place) all parameter values in scope with the values contained in the file.

        Paramters
        ---------
        filepath : str
        """
        with open(filepath, "rb") as f:
            params = pickle.load(f)
        for p, value in params["parameters"].items():
            self._parameters[p].value = value
        for name, state_dict in params["models"].items():
            model = self._namespace[name]
            model.load_state_dict(state_dict)

    def clamp_parameters(self):
        """Update the value of all parameters in-place to satisfy the constraints of their domain."""
        logger.info("Clamping parameters.")
        with torch.no_grad():
            for _, parameter in self._parameters.items():
                parameter.clamp()

    def parameters(self):
        """Return an iterable over all tuneable parameters in-scope.

        Returns
        -------
        Iterable[Tensor]
        """
        for _, parameter in self._parameters.items():
            yield parameter.value
        for _, obj in self._namespace.items():
            if hasattr(obj, "parameters"):
                yield from obj.parameters()
        yield from self.posterior.parameters()

    @property
    def parameter_map(self):
        return {n : p.value for n, p in self._parameters.items()}