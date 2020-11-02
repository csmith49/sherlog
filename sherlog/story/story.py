from collections import defaultdict
import torch
import pyro
from pyro.infer.predictive import Predictive
from .statement import Statement
from .observation import Observation
from .context import Context, Value

class Story:
    def __init__(self, statements, observations, parameters, namespaces):
        '''Stories are combinations of generative procedures and observations on the generated values.

        Parameters
        ----------
        statements : Statement list

        observations : Observation list

        parameters : (string, Parameter) dict

        namespaces : (string, Namespace) dict
        '''
        self.statements = statements
        self.observations = observations
        self.parameters = parameters
        self.namespaces = namespaces

        self.variable_indices = {s.variable.name : i for i, s in enumerate(self.statements)}

        # map variable indices to variable indices
        self.dependency_graph = defaultdict(lambda: [])
        self.dataflow_graph = defaultdict(lambda: [])

        for stmt in self.statements:
            source = self.variable_indices[stmt.variable.name]
            for dependency in stmt.dependencies:
                dest = self.variable_indices[dependency.name]
                self.dependency_graph[source].append(dest)

        for dest, sources in self.dependency_graph.items():
            for source in sources:
                self.dataflow_graph[source].append(dest)

    def variables(self):
        '''Returns all variables that statement executions can get bound to.

        Returns
        -------
        string iterator
        '''
        for stmt in self.statements:
            yield stmt.variable.name

    def dataflow(self, stmt):
        '''Computes all statements that directly depend on the output of the given statement.

        Parameters
        ----------
        stmt : Statement

        Returns
        -------
        Statement iterator
        '''
        index = self.variable_indices[stmt.variable.name]
        for destination in self.dataflow_graph[index]:
            yield self.statements[destination]

    def __str__(self):
        return '\n'.join(str(stmt) for stmt in self.topological_statements())

    def context(self):
        '''Builds a fresh context from the instance parameters and namespaces.

        Returns
        -------
        Context
        '''
        return Context(self.parameters, self.namespaces)

    @classmethod
    def of_json(cls, statements, observations, parameters, namespaces):
        '''Builds a story from a JSON-like object and an already-parsed context (in the form of `parameters` and `namespaces`).

        Parameters
        ----------
        statements : (JSON-like object) list

        observations : (JSON-like object) list

        parameters : (string, Paramter) dict

        namespaces : (string, Namespace) dict
        '''
        statements = [Statement.of_json(s) for s in statements]
        observations = [Observation.of_json(o) for o in observations]
        return cls(statements, observations, parameters, namespaces)

    def topological_statements(self):
        '''Iterates over the statements in the story in topological order.

        The partial order is derived from the dataflow relationship.

        Returns
        -------
        Statement iterator
        '''
        # instead of removing edges, we just track which variables have been resolved
        resolved_variables = set()
        def is_resolved(stmt):
            return all(v.name in resolved_variables for v in stmt.dependencies)
        # collect all statements with no parents in a list
        initial_statements = [s for s in self.statements if is_resolved(s)]
        # iteratively mark edges from initial statements as "resolved"
        while initial_statements:
            # if it's in the initial list, we've already handled the deps
            stmt = initial_statements.pop()
            yield stmt
            # extend initial with any dataflow children that are now resolved
            resolved_variables.add(stmt.variable.name)
            initial_statements += [s for s in self.dataflow(stmt) if is_resolved(s)]

    def run(self):
        '''Executes the story.

        Returns
        -------
        Context
        '''
        context = self.context()
        for statement in self.topological_statements():
            name, value = statement.run(context)
            context[name] = value
        return context

    def loss(self):
        '''A function that - when differentiated - gives the gradient of the parameters wrt the Viterbi loss.

        Returns
        -------
        tensor
        '''
        context = self.run()
        return surrogate_viterbi_loss(self.observations, context)

    def pyro_model(self):
        '''Builds a Pyro model for the generative process.

        Returns
        -------
        Context
        '''
        context = self.run()
        # build the site for the observations
        distances = [
            observation_distance(obs, context) for obs in self.observations
        ]
        value = min(distances)
        distribution = pyro.deterministic("sherlog:result", value)
        log_probs = stochastic_dependencies(self.observations, context)
        context["sherlog:result"] = Value(value, distribution, log_probs)
        return context

    def likelihood(self, num_samples=100):
        '''Empirically computes the likelihood that the story produces results consistent with the observations.

        Parameters
        ----------
        num_samples : int (default 100)

        Returns
        -------
        1-d tensor
        '''
        predictive = Predictive(self.pyro_model, num_samples=num_samples, return_sites=("sherlog:result",))
        results = predictive()["sherlog:result"]
        return torch.sum(results, dim=0) / num_samples

def observation_distance(observation, context, p=1):
    '''Computes the distance from an observation to the values in the context.

    Parameters
    ----------
    observation : Observation

    context : Context

    p : int (default 1)

    Returns
    -------
    tensor
    '''
    distance = torch.tensor(0.0)
    for name, value in observation.evaluate(context):
        distance += torch.dist(value.value, context[name].value, p=1) ** p
    return distance ** (1/p)

def stochastic_dependencies(observations, context):
    '''Computes the log-probs for all values in the observations.

    Parameters
    ----------
    observations : Observation list

    context : Context

    Returns
    -------
    (string, tensor) dict
    '''
    dependencies = {}
    for observation in observations:
        for name, _ in observation.items():
            deps = context[name].log_probs
            dependencies = {**dependencies, **deps}
    return dependencies

def surrogate_viterbi_loss(observations, context):
    '''DiCE-based surrogate for the Viterbi loss.

    Parameters
    ----------
    observations : Observation list

    context : Context

    Returns
    -------
    tensor
    '''
    distances = torch.tensor([
        observation_distance(obs, context) for obs in observations
    ])
    dependencies = stochastic_dependencies(observations, context)
    log_probs = [v for _, v in dependencies.items()]
    tau = sum(log_probs, torch.tensor(0.0))
    magic_box = torch.exp(tau - tau.detach())
    cost = torch.min(distances, dim=0).values
    return magic_box * cost
