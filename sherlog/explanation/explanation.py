"""Explanation."""

from sherlog.program import parameter
from ..engine import Store, value
from .observation import Observation
from ..engine import Model
from .history import History
from ..logs import get
from . import semantics
import torch

logger = get("explanation")

# TODO - fix this, don't think it's right
def log_sub(x, y):
    return x + (y - x).exp().mul(-1).log1p()
    return max(x, y) + (x - y).abs().mul(-1).exp().log1p()


class Explanation:
    """Explanations are the central explanatory element in Sherlog. They capture sufficient generative properties to ensure a particular outcome."""

    def __init__(self, model, meet, avoids, meet_history, avoid_histories, external=()):
        """Constructs an explanation.

        Parameters
        ----------
        model : Model
        meet : Observation
        avoids : List[Observation]
        meet_history : History
        avoid_history : List[History]
        external : Mapping[str, Any] (default={})
        """
        logger.info(f"Explanation {self} built.")
        self.model = model
        self.meet = meet
        self.avoids = avoids
        self.meet_history = meet_history
        self.avoid_histories = avoid_histories
        self._external = external

    @classmethod
    def of_json(cls, json, external=()):
        """Constructs an explanation from a JSON encoding.

        Parameters
        ----------
        json : JSON-like object
        external : Mapping[str, Any] (default={})

        Returns
        -------
        Explanation
        """
        logger.info(f"Building explanation from serialization: {json}...")
        model = Model.of_json(json["assignments"])
        meet = Observation.of_json(json["meet"])
        avoids = [Observation.of_json(obs) for obs in json["avoid"]]
        meet_history = History.of_json(json["meet_history"])
        avoid_histories = [History.of_json(h) for h in json["avoid_history"]]
        return cls(model, meet, avoids, meet_history, avoid_histories, external=external)

    def positive(self):
        """Positive aspect of the explanation.

        Returns
        -------
        Tuple[Observation, History]
        """
        return self.meet, self.meet_history
    
    def negatives(self):
        """Negative aspects of the explanation.

        Returns
        -------
        Iterable[Tuple[Observation, History]]
        """
        for avoid, avoid_history in zip(self.avoids, self.avoid_histories):
            yield self.meet + avoid, self.meet_history + avoid_history

    @property
    def store(self):
        """Store initialized with external mapping.

        Returns
        -------
        Store
        """
        return Store(external=self._external)

    def run(self, functor, wrap_args={}, fmap_args={}, parameters={}):
        """Evaluate the explanation in the given functor.

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
        """Use provided functor to evaluate the explanation and build optimization objective.

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
        avoids = [obs.equality(store, functor, prefix=f"sherlog:avoid:{i}", default=0.0) for i, obs in enumerate(self.avoids)]
        avoid = value.Variable("sherlog:avoid")
        if avoids:
            functor.run(avoid, "or", avoids, store)
        else:
            functor.run(avoid, "set", [0.0], store)

        # build objective
        objective = value.Variable("sherlog:objective")
        functor.run(objective, "satisfy", [meet, avoid], store)

        # just return the computed objective - functor should track all relevant info
        return store[objective]

    def miser(self, samples=1):
        """Build a Miser surrogate objective for the story.

        Parameters
        ----------
        samples : int (default=1)
            Number of simultaneous executions to perform.

        Returns
        -------
        Tensor
        """

        logger.info("Evaluating types and forcing values...")
        # build type info for the forcing
        # types = self.run(semantics.types.functor)
        forcing = {}

        # add the values from meet
        for variable in self.meet.variables:
            forcing[variable] = self.meet[variable]

        # and, if possible, add values from avoid
        # for avoid in self.avoids:
        #     for variable in avoid.variables:
        #         if types[variable] == semantics.types.Discrete(2):
        #             forcing[variable] = 1 - avoid[variable]

        logger.info(f"Forcing with observations: {forcing}")

        # build the miser functor with the computed forcings
        functor = semantics.miser.factory(samples, forcing=forcing)
        objective = self.objective(functor)

        # build surrogate
        scale = semantics.miser.forcing_scale(objective.dependencies()).detach() # if we don't detach here, we "overcount" forced instances
        score = semantics.miser.magic_box(objective.dependencies(), samples)
        surrogate = objective.value * scale * score

        logger.info(f"Objective, likelihood ratio, and magic box: {objective} / {scale} / {score}")
        logger.info(f"Miser surrogate objective: {surrogate}")

        return surrogate

    def graph(self):
        """Build a graph representation of the story.

        Returns
        -------
        networkx.DiGraph
        """
        objective = self.objective(semantics.graph.functor)
        return semantics.graph.to_graph(objective)

    def objective(self, functor, observation):
        """Build computation graph for the satisfaction of the given observation in the given functor.

        Parameters
        ----------
        functor : Functor
        observation : Observation

        Returns
        -------
        Functor.t
        """
        store = self.run(functor)

        obs = observation.equality(store, functor, prefix="sherlog:observation", default=1.0)
        objective = value.Variable("sherlog:objective")
        functor.run(objective, "set", [obs], store)

        return store[objective]

    def miser(self, observation, samples=1):
        """Builds a Miser surrogate objective for the satisfaction of the given observation.

        Parameters
        ----------
        observation : Observation
        samples : int (default=1)

        Returns
        -------
        Miser.t (with batch dimension of size `samples`)
        """
        # force all the observed variables
        forcing = {}
        for variable in observation.variables:
            forcing[variable] = observation[variable]

        # construct the objective
        functor = semantics.miser.factory(samples, forcing=forcing)
        objective = self.objective(functor, observation)

        # scale and score appropriately
        scale = semantics.miser.forcing_scale(objective.dependencies()).detach() # if we don't detach, we seem to "overcount" forced instances
        score = semantics.miser.magic_box(objective.dependencies(), samples)
        surrogate = objective.value * scale * score

        return surrogate

    def log_prob(self, parameterization, samples=1):
        # build negative info
        neg_exp = []
        for o, h in self.negatives():
            lp = self.miser(o, samples=samples).mean().log() - h.log_prob(parameterization)
            neg_exp.append(lp)
        
        if neg_exp:
            n = torch.mean(torch.stack(neg_exp))
        else:
            n = torch.tensor(0.0).log()

        # build positive info
        o, h = self.positive()
        p = self.miser(o, samples=samples).mean().log()
        q = h.log_prob(parameterization)

        # and combine it all together
        result = log_sub(p, n) - q
        print(p, n, q, result)
        return result