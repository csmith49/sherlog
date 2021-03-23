from ..engine import Store, value
from ..inference import Objective
from .observation import Observation
from ..engine import Model
from ..logs import get
from . import semantics

import torch
import storch

logger = get("story")

class Story:
    def __init__(self, model, meet, avoid, external=()):
        logger.info(f"Story {self} built.")
        self.model = model
        self.meet = meet
        self.avoid = avoid
        self._external = external

    @classmethod
    def of_json(cls, json, external=()):
        logger.info(f"Building story from serialization: {json}...")
        model = Model.of_json(json["assignments"])
        meet = Observation.of_json(json["meet"])
        avoid = Observation.of_json(json["avoid"])
        return cls(model, meet, avoid, external=external)

    @property
    def store(self):
        return Store(external=self._external)

    def run(self, functor, wrap_args={}, fmap_args={}, parameters={}):
        """Evaluate the story in the given functor.

        Parameters
        ----------
        functor : Functor

        wrap_args : Optional[Dict[string, Any]]

        fmap_args : Optional[Dict[string, Any]]

        parameters : Optional[Dict[string, Dict[string, Any]]]

        Returns
        -------
        Store[Functor.t]

        """
        store = self.store
        for assignment in self.model.assignments:
            functor.run_assignment(
                assignment, 
                store, 
                wrap_args=wrap_args, 
                fmap_args=fmap_args, 
                parameters=parameters)
        return store

    def objective(self, functor):
        """Use provided functor to evaluate the story and build optimization objective.

        Parameters
        ----------
        functor : Functor

        Returns
        -------
        Functor.t
        """
        # get values
        store = self.run(functor)

        # build meet and avoid
        meet = self.meet.equality(store, functor, prefix="sherlog:meet", default=1.0)
        avoid = self.avoid.equality(store, functor, prefix="sherlog:avoid", default=0.0)

        # build objective
        objective = value.Variable("sherlog:objective")
        functor.run(objective, "satisfy", [meet, avoid], store)

        # just return the computed objective - functor should track all relevant info
        return store[objective]

    def dice(self):
        """Build DiCE surrogate objective for a single execution of the story.

        Returns
        -------
        Tensor
        """
        # use dice functor to build surrogate objective
        objective = self.objective(semantics.dice.functor)
        score = semantics.dice.magic_box(*objective.dependencies())
        surrogate = objective.value * score

        # check to make sure gradients are being passed appropriately
        if surrogate.grad_fn is None:
            logger.warning(f"DiCE objective {surrogate} has no gradient.")

        return surrogate

    def shaped(self):
        """Build shaped reward for a single execution of the story.

        Returns
        -------
        Tensor
        """
        # use dice functor to build surrogate objective
        objective = self.objective(semantics.shaped.functor)
        score = semantics.shaped.magic_box(*objective.dependencies())
        surrogate = objective.value * score

        # check to make sure gradients are being passed appropriately
        if surrogate.grad_fn is None:
            logger.warning(f"SHaped reward objective {surrogate} has no gradient.")

        return surrogate

    def forced(self):
        """Build build objective for a forced execution of the story.

        Returns
        -------
        Tensor
        """
        functor = semantics.forced.functor(self.meet)
        store = self.run(functor)

        # build meet and avoid
        meet = self.meet.equality(store, functor, prefix="sherlog:meet", default=1.0)
        avoid = self.avoid.equality(store, functor, prefix="sherlog:avoid", default=0.0)

        # build objective
        objective = value.Variable("sherlog:objective")
        functor.run(objective, "satisfy", [meet, avoid], store)

        # also need to build scale from forced observations
        if self.meet.is_empty:
            scale = torch.tensor(1.0)
        else:
            probs = torch.stack([store[f].log_prob for f in self.meet.variables])
            scale = torch.exp(torch.sum(probs))
        
        # build the score - likelihood of execution
        score = semantics.forced.magic_box(*store[objective].dependencies())
        
        # and construct the surrogate
        surrogate = torch.log(store[objective].value * scale * score)

        # check to make sure gradients are being passed appropriately
        if surrogate.grad_fn is None:
            logger.warning(f"SHaped reward objective {surrogate} has no gradient.")

        return surrogate

    def graph(self):
        """Build a graph representation of the story.

        Returns
        -------
        networkx.DiGraph
        """
        objective = self.objective(semantics.graph.functor)
        return semantics.graph.to_graph(objective)

    def likelihood(self, samples=1):
        """Estimate the likelihood."

        Parameters
        ----------
        samples : int
            Defaults to 1.

        Returns
        -------
        Tensor
        """
        scores = [self.objective(semantics.tensor.functor) for _ in range(samples)]
        return torch.mean(torch.tensor(scores))