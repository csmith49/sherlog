from .parameter import Parameter
from .evidence import Evidence
from .posterior import LinearPosterior
from ..engine import Store

from ..explanation import Explanation
from ..interface import query
from ..logs import get

from typing import Iterable, Any, Dict, Optional
from torch import tensor, Tensor, stack, no_grad
from itertools import islice

import pickle
import random

logger = get("program")

class Program:
    """Programs coordinate generation of explanations."""

    def __init__(self, source, parameters : Iterable[Parameter], namespace : Dict[str, Any], contexts : Iterable[str] = ()):
        """Construct a program from source.
        
        Parameters
        ----------
        source : JSON-like object

        parameters : Iterable[Parameter]

        namespace : Dict[str, Any]

        contexts : Iterable[str] (default=())
        """
        # convert parameters to map
        self._parameters = {parameter.name : parameter for parameter in parameters}
        self._namespace = namespace
        self._source = source
        self._posterior = LinearPosterior(contexts=contexts)

    @classmethod
    def of_json(cls, json, namespace : Optional[Dict[str, Any]] = None) -> 'Program':
        """Build a program from a JSON-like object.
        
        Parameters
        ----------
        json : JSON-like objectt
        
        namespace : Optional[Dict[str, Any]]

        Returns
        -------
        Program
        """
        parameters = [Parameter.of_json(parameter) for parameter in json["parameters"]]
        source = {
            "rules" : json["rules"],
            "parameters" : json["parameters"],
            "ontology" : json["ontology"],
            # evidence split out elsewhere, but server expects this
            "evidence" : []
        }
        namespace = namespace if namespace else {}
        # pull together
        return cls(source, parameters, namespace)

    def explanations(self, evidence : Evidence, quantity : int = 1, attempts : int = 100, width : Optional[int] = None) -> Iterable[Explanation]:
        """Sample explanations for the provided evidence.

        Parameters
        ----------
        evidence : Evidence

        quantity : int (default=1)

        attempts : int (default=100)

        width : Optional[int]

        Returns
        -------
        Iterable[Explanation]
        """
        logger.info("Sampling explanations for evidence: %s...", evidence)

        # build kwargs for queries
        kwargs = {}
        kwargs["width"] = width
        kwargs["contexts"] = list(self._posterior.contexts)
        kwargs["parameterization"] = self._posterior.parameterization()

        # build generator
        def gen():
            for attempt in range(attempts):
                logger.info("Starting explanation generation attempt %i...", attempt)
                try:
                    for json in query(self._source, evidence.json, **kwargs):
                        yield Explanation.of_json(json)
                except TimeoutError:
                    logger.warning("Explanation generation attempt %i timed out. Restarting...", attempt)

        # get at most quantity explanations
        yield from islice(gen(), quantity)

    def store(self, **locals) -> Store:
        """Construct a store for evaluating explanations of the program.
        
        Parameters
        ----------
        **locals
            Any extra bindings to be added to the store.
        
        Returns
        -------
        Store
        """
        return Store(**self._parameters, **self._namespace, **locals)

    def log_prob(self, evidence : Evidence, explanations : int = 1, attempts : int = 100, width : Optional[int] = None, **locals) -> Tensor:
        """Compute the marginal log-likelihood of the provided evidence.
        
        Parameters
        ----------
        evidence : Evidence

        explanations : int (default=1)

        attempts : int (default=100)

        width : Optional[int]

        **locals
            Local bindings to pass to the explanation during execution.

        Returns
        -------
        Tensor
        """
        store = self.store(**locals)
        exs = self.explanations(evidence, quantity=explanations, width=width, attempts=attempts)
        log_probs = [ex.log_prob(store) - self._posterior.log_prob(ex) for ex in exs]
        # if no explanations, default
        if log_probs:
            return stack(log_probs).mean()
        else:
            return tensor(0.0)

    def parameters(self, **locals) -> Iterable[Tensor]:
        """Returns all tuneable parameters in the program and namespace, if provided.
        
        Parameters
        ----------
        **locals
            Local bindings to explore.

        Returns
        -------
        Iterable[Tensor]
        """
        # handle parameters
        for _, parameter in self._parameters.items():
            yield parameter.value
        # handle internal namespace
        for _, obj in self._namespace.items():
            if hasattr(obj, "parameters"):
                yield from obj.parameters()
        # handle external namespace
        for _, obj in locals.items():
            if hasattr(obj, "parameters"):
                yield from obj.parameters()
        # handle posterior
        yield from self._posterior.parameters()

    def clamp(self):
        """Update parameters in-place to satisfy the constraints of their domain."""
        logger.info("Clamping parameters...")
        with no_grad():
            for _, parameter in self._parameters.items():
                parameter.clamp()

    def sample_explanation(self, evidence : Evidence, burn_in : int = 100, namespace : Optional[Dict[str, Any]] = None, **kwargs) -> Explanation:
        """Sample an explanation from the posterior.

        Parameters
        ----------
        evidence : Evidence
        
        burn_in : int (default=100)

        namespace : Optional[Dict[str, Any]]

        **kwargs
            Passed to explanation generation during execution.

        Returns
        -------
        Explanation
        """
        logger.info(f"Sampling explanation for {evidence} with {burn_in} burn-in steps.")
        
        sample, sample_likelihood = None, 0.00001

        for step in range(burn_in):
            # sample a new explanation and compute likelihood
            explanation = next(self.explanations(evidence, quantity=1, **kwargs))
            explanation_likelihood = explanation.log_prob(self._posterior.parameterization, namespace=namespace).exp()

            # accept/reject
            ratio = explanation_likelihood / sample_likelihood
            if random.random() <= ratio:
                logger.info(f"Step {step}: sample accepted with likelihood ratio {ratio}.")
                sample, sample_likelihood = explanation, explanation_likelihood

        if sample is None:
            logger.warning(f"No sample accepted after {burn_in} burn-in steps.")
        
        return sample

    # SAVING AND LOADING
    # probably broken wrt external models...
    # oh, and no posterior saved

    def save_parameters(self, filepath : str):
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

    def load_parameters(self, filepath : str):
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
