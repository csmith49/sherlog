from collections import defaultdict
import torch
from . import statement
from . import observation

class Story:
    def __init__(self, statements, observations, parameters, functions):
        self.statements = statements
        self.observations = observations
        self.parameters = parameters
        self.functions = functions

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
        for stmt in self.statements:
            yield stmt.variable.name

    def dataflow(self, stmt):
        index = self.variable_indices[stmt.variable.name]
        for destination in self.dataflow_graph[index]:
            yield self.statements[destination]

    def __str__(self):
        return "\n".join(str(stmt) for stmt in self.statements)

    def trainable_parameters(self):
        for _, parameter in self.parameters.items():
            yield parameter.value
        for _, function in self.functions.items():
            if hasattr(function, "parameters"):
                yield from function.parameters()

    @classmethod
    def of_json(cls, statements, observations, parameters, functions):
        statements = [statement.of_json(s) for s in statements]
        observations = [observation.of_json(o) for o in observations]
        return cls(statements, observations, parameters, functions)

    def topological_statements(self):
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

    def run(self, parameters):
        namespace = {k : v for k, v in parameters.items()}
        for statement in self.topological_statements():
            variable, value = statement.run(namespace, self.functions)
            namespace[variable] = value
        return namespace

    def dice_run(self):
        namespace = {k : v.value for k, v in self.parameters.items()}
        log_probs = {}
        for statement in self.topological_statements():
            variable, value, log_prob = statement.dice(namespace, self.functions)
            namespace[variable] = value
            if log_prob is not None:
                log_probs[variable] = log_prob
        return namespace, log_probs

    def cost(self, namespace, observations):
        distances = []
        for ob in observations:
            distance = torch.tensor(0.0)
            for var, value in ob.items():
                distance += torch.dist(torch.tensor(value), namespace[var])
            distances.append(distance)
        # approximate the smallest distance
        if len(distances) == 1: return distances[0]
        else:
            return -1.0 * torch.softmax(*[-1.0 * d for d in distances])

    def loss(self, observations):
        namespace, log_probs = self.dice_run()
        cost = self.cost(namespace, observations)
        # dice objective computation
        tau = sum(p for v, p in log_probs.items())
        magic_box = torch.exp(tau - tau.detach())
        objective = magic_box * cost
        return objective