from collections import defaultdict
from itertools import chain
import torch
import pyro
from pyro.infer.predictive import Predictive
from torch import is_storage
from .statement import Statement
from .observation import Observation
from .context import Context
from . import semantics

def magic_box(log_probs):
    tau = sum(log_probs, start=torch.tensor(0.0))
    return torch.exp(tau - tau.detach())

def viterbi_objective(observations, context):
    '''Objective that - when minimized - maximizes the Viterbi evidence of the observations.

    Parameters
    ----------
    observations : Observation list

    context : Context

    Returns
    -------
    (tensor, string list)
    '''
    distances = [obs.distance(context).reshape(1) for obs in observations]
    variables = chain(*(obs.variables() for obs in observations))
    return torch.min(torch.cat(distances), dim=0).values, variables

class Story:
    def __init__(self, statements, observations):
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

        self.variable_indices = {s.variable.name : i for i, s in enumerate(self.statements)}
        self.variable_indices_inverse = {v : k for k, v in self.variable_indices.items()}

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

    def dependencies(self, *variables):
        '''Computes the variable dependencies of a provided set of variables.

        Parameters
        ----------
        variables : string iterator

        Returns
        -------
        string iterator
        '''
        visited = set()
        worklist = [self.variable_indices[var] for var in variables]
        while worklist:
            var = worklist.pop()
            if var not in visited:
                visited.add(var)
                worklist.extend(self.dependency_graph[var])
        return [self.variable_indices_inverse[i] for i in visited]

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

    @classmethod
    def of_json(cls, statements, observations):
        '''Builds a story from a JSON-like object.

        Parameters
        ----------
        statements : (JSON-like object) list

        observations : (JSON-like object) list

        Returns
        -------
        Story
        '''
        statements = [Statement.of_json(s) for s in statements]
        observations = [Observation.of_json(o) for o in observations]
        return cls(statements, observations)

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

    def run(self, context, semantics, **kwargs):
        '''Executes the story over the provided context, returning a fresh context with the execution stored.

        Parameters
        ----------
        context : Context

        Returns
        -------
        Context
        '''
        context = context.clone()
        for statement in self.topological_statements():
            name, value = statement.run(context, semantics, **kwargs)
            context[name] = value
        return context

    def pyro_model(self, context):
        '''Builds a Pyro model for the generative process executed over the context.

        Parameters
        ----------
        context : Context

        Returns
        -------
        Context
        '''
        context = self.run(context, semantics.pyro.run)
        # build the site for the observations
        similarities = [
            obs.similarity(context) for obs in self.observations
        ]
        context["sherlog:result"] = pyro.deterministic("sherlog:result", max(similarities))
        return context

    def likelihood(self, context, offset=1, num_samples=100):
        '''Empirically computes the likelihood that the story produces results consistent with the observations.

        Parameters
        ----------
        context : Context

        offset : int (default 1)

        num_samples : int (default 100)

        Returns
        -------
        1-d tensor
        '''
        predictive = Predictive(self.pyro_model, num_samples=num_samples, return_sites=("sherlog:result",))
        results = predictive(context)["sherlog:result"]
        return (offset + torch.sum(results, dim=0)) / (offset + num_samples)

    def loss(self, context, objectives=(viterbi_objective,), **kwargs):
        '''A function that - when differentiated - gives the gradient of the parameters wrt the sum of the objectives.

        Parameters
        ----------
        context : Context

        objectives : objective iterable

        Returns
        -------
        tensor
        '''
        context = self.run(context, semantics.dice.run, **kwargs)
        # for each objective, compute the cost and the log_probs of all stochastic dependencies
        cost_nodes = []
        for obj in objectives:
            cost, vars_used = obj(self.observations, context)
            log_probs = []
            for dep in self.dependencies(*vars_used):
                if context[dep].is_stochastic:
                    log_probs.append(context[dep].log_prob)
            cost_nodes.append( (cost, log_probs) )

        # build the standard surrogate
        surrogate = torch.tensor(0.0)
        for (cost, log_probs) in cost_nodes:
            mb_c = magic_box(log_probs)
            surrogate += mb_c * cost

        return surrogate
