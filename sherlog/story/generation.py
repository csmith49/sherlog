from . import term
import torch.distributions as dist

class UnknownGeneration(Exception):
    def __init__(self, generation):
        """Error representing the provided generation is not recognized.

        Parameters
        ----------
        generation : string
            Name of the generation that is not recognized by the translation process
        """
        self.generation = generation

    def __str__(self):
        return f"Generation {self.generation} has no translation"

class Generation:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

    def __str__(self):
        return f"{self.name}[{', '.join(str(arg) for arg in self.arguments)}]"

    @classmethod
    def of_json(cls, json):
        name = json["function"]
        arguments = [term.of_json(v) for v in json["arguments"]]
        return cls(name, arguments)

    def evaluate_arguments(self, namespace):
        arguments = []
        for arg in self.arguments:
            if isinstance(arg, term.Variable):
                arguments.append(namespace[arg.name])
            elif isinstance(arg, str):
                arguments.append(namespace[arg])
            else:
                arguments.append(arg)
        return arguments

    def to_torch(self, namespace, functions):
        arguments = self.evaluate_arguments(namespace)
        # case-by-case translation of distributions
        if self.name == "beta":
            alpha, beta = arguments[0], arguments[1]
            return dist.Beta(alpha, beta).rsample(), None
        elif self.name == "normal":
            mean, sdev = arguments[0], arguments[1]
            return dist.Normal(mean, sdev).rsample(), None
        elif self.name == "bernoulli":
            success = arguments[0]
            distribution = dist.Bernoulli(success)
            result = distribution.sample()
            return result, distribution.log_prob(result)
        # try to find a function implementation
        else:
            try:
                f = functions[self.name]
                return f(*arguments), None
            except KeyError:
                pass
        raise UnknownGeneration(self.name)

def of_json(json):
    return Generation.of_json(json)