from .namespace import Namespace
from .parameter import Parameter
from .evidence import Evidence
from ..inference import Objective
from .. import interface
from ..engine import Model, value, Store
from ..story import Story
from ..logs import get

from itertools import islice, chain, repeat

import torch
import pickle
import random

logger = get("problem")

class Problem:
    def __init__(self, parameters, namespaces=None, evidence=None, program=None):
        """Object representing a Sherlog problem file.

        Parameters
        ----------
        parameters : Iterable[Parameter]

        namespaces : Iterable[str]

        evidence : Iterable[Evidence]

        program : JSON
        """
        # convert params to name-param map
        self._parameters = {p.name : p for p in parameters}
        # build the namespace obj from the namespaces provided
        self._namespace = Namespace(modules=namespaces)
        # just record the rest
        self._evidence = evidence
        self.program = program

    @property
    def evidence(self):
        yield from self._evidence

    @classmethod
    def of_json(cls, json):
        """Build a problem from a JSON-like object.

        Parameters
        ----------
        json : JSON
        
        Returns
        -------
        Problem
        """
        parameters = [Parameter.of_json(p) for p in json["parameters"]]
        # namespaces = json["namespaces"]
        evidence = [Evidence.of_json(e) for e in json["evidence"]]
        program = json
        return cls(parameters=parameters, namespaces=[], evidence=evidence, program=program)

    def stories(self, evidence, samples=1, attempts=100, width=None, depth=None):
        """Construct all stories encoded by the problem.

        Parameters
        ----------
        evidence : Evidence

        samples : int
            Number of samples desired (defaults to 1).

        attempts : int
            Maximum number of queries to logic server (defaults to 100).

        width : Optional[int]
            Maximum beam width during stochastic resolution.
        
        depth : Optional[int]
            Maximum proof depth.

        Returns
        -------
        Iterable[List[Story]]
        """

        logger.info(f"Sampling stories for evidence {evidence}...")

        # build the external evaluation context with the namespace
        external = (self.parameter_map, self._namespace)

        # build the generator
        def gen():
            for attempt in range(attempts):
                logger.info(f"Starting story generation attempt {attempt}...")
                for json in interface.query(self.program, evidence.json, width=width, depth=depth):
                    logger.info("Story found.")
                    yield Story.of_json(json, external=external)
        
        # and grab only the number of samples desired
        yield from islice(gen(), samples)

    def marginal_likelihood(self, evidence, stories=1, samples=1):
        """Compute the marginal likelihood of a piece of evidence.

        Parameters
        ----------
        evidence : Evidence

        stories : int
            Number of stories to sample (defaults to 1).

        samples : int
            Number of samples of likelihood to compute per-story (defaults to 1).

        Returns
        -------
        Tensor
        """
        story_iter = self.stories(evidence, samples=stories)
        samples = torch.cat([story.miser(samples=samples) for story in story_iter])

        likelihood = torch.mean(samples)

        logger.info(f"Evidence {evidence} has likelihood {likelihood:f} with variance {samples.var()}.")

        # give a warning if we've somehow constructed value with no gradient
        if likelihood.grad_fn is None:
            logger.warning(f"Evidence {evidence} has likelihood {likelihood} with no gradient.")

        return likelihood

    def objectives(self, epoch=None, stories=1, samples=1):
        """Generates log-likelihood objectives for all evidence.

        Parameters
        ----------
        epoch : Optional[int]
            Current epoch.

        stories : int

        samples : int

        Returns
        -------
        Iterable[Objective]
        """
        # build the objective header using kwargs
        if epoch is not None:
            HEADER = f"{epoch}:ll"
        else:
            HEADER = "ll"

        # yield objective per-evidence
        for evidence in self.evidence:
            likelihood = self.marginal_likelihood(evidence, stories=stories, samples=samples)
            obj = Objective(f"{HEADER}:{evidence}", torch.log(likelihood))
            yield obj

    def log_likelihood(self, stories=1, samples=1):
        """Compute the log-likelihood of the problem.

        Parameters
        ----------
        stories : int (default=1)
        samples : int (default=1)

        Returns
        -------
        Tensor
        """
        marginal = lambda e: self.marginal_likelihood(e, stories=stories, samples=samples)
        marginals = torch.stack([marginal(evidence) for evidence in self.evidence])
        return marginals.log().sum()

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

    @property
    def parameter_map(self):
        return {n : p.value for n, p in self._parameters.items()}