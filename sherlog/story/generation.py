from . import term
from .context import Value
import torch.distributions as torch_dist
import pyro
import pyro.distributions as pyro_dist

class UnknownGeneration(Exception):
    def __init__(self, generation):
        """Error representing the provided generation is not recognized.

        Parameters
        ----------
        generation : Generation
            The generation that is not recognized by the translation process
        """
        self.generation = generation

    def __str__(self):
        return f"Generation mechanism {self.generation} has no translation"

class Generation:
    def __init__(self, name, arguments):
        '''Generations are the mechanisms that produce values, either stochastically or deterministically.

        Parameters
        ----------
        name : string

        arguments : Term list
        '''
        self.name = name
        self.arguments = arguments

    def __str__(self):
        return f"{self.name}[{', '.join(str(arg) for arg in self.arguments)}]"

    @classmethod
    def of_json(cls, json):
        '''Converts a JSON-like object to a Generation.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Generation
        '''
        name = json["function"]
        arguments = [term.of_json(v) for v in json["arguments"]]
        return cls(name, arguments)

    def evaluate_arguments(self, context):
        '''Evaluates the list of arguments to Value objects.

        Parameters
        ----------
        context : Context

        Returns
        -------
        Value list
        '''
        arguments = []
        for arg in self.arguments:
            # if we have a variable, look it up in the context
            if isinstance(arg, term.Variable):
                arguments.append(context[arg.name])
            # same if we have a string - that's a symbolic value we need to concretize
            elif isinstance(arg, str):
                arguments.append(context[arg])
            # otherwise, we're some object fresh from the parser and need to be lifted
            else:
                value = Value.lift(arg)
                arguments.append(value)
        return arguments

    def evaluate(self, name, context):
        '''Evaluates the generation mechanism in context `context`.

        An identifier is also passed to bind the constructed Pyro sample site.

        Parameters
        ----------
        name : string

        context : Context

        Returns
        -------
        Value
        '''
        arguments = self.evaluate_arguments(context)
        # case-by-case translation
        
        # reparameterized beta distribution
        if self.name == "beta":
            a, b = arguments[0], arguments[1]
            return beta(name, a, b)
    
        # reparameterized normal distribution
        elif self.name == "normal":
            mean, sdev = arguments[0], arguments[1]
            return normal(name, mean, sdev)
        
        # standard bernoulli distribution
        elif self.name == "bernoulli":
            success = arguments[0]
            return bernoulli(name, success)

        # check external functions
        else:
            try:
                f = self.context.lookup_callable(self.name)
                return external(name, f, *arguments)
            except KeyError: pass

        # if none of the above work, raise an exception
        raise UnknownGeneration(self)

# ======== GENERATIVE MECHANISMS ========

def normal(name, mean, sdev):
    '''A reparameterized normal distribution with parameters `mean` and `sdev`.

    Parameters
    ----------
    name : string

    mean : Value

    sdev : Value

    Returns
    -------
    Value
    '''
    value = torch_dist.Normal(mean.value, sdev.value).rsample()
    distribution = pyro.sample(name, pyro_dist.Normal(mean.distribution, sdev.distribution))
    return Value(value, distribution)

def beta(name, alpha, beta):
    '''A reparameterized Beta distribution with parameters `alpha` and `beta`.

    Parameters
    ----------
    name : string

    alpha : Value

    beta : Value

    Returns
    -------
    Value
    '''
    value = torch_dist.Beta(alpha.value, beta.value).rsample()
    distribution = pyro.sample(name, pyro_dist.Beta(alpha.distribution, beta.distribution))
    return Value(value, distribution)

def bernoulli(name, success):
    '''A Bernoulli distribution with parameter `success`.

    Parameters
    ----------
    name : string

    success : Value

    Returns
    -------
    Value
    '''
    value = torch_dist.Bernoulli(success.value).sample()
    distribution = pyro.sample(name, pyro_dist.Bernoulli(success.distribution))
    log_prob = torch_dist.Bernoulli(success.value).log_prob(value)
    return Value(value, distribution, log_prob=log_prob)

def external(name, callable, *args):
    '''A call to an external callable with parameters `args`.

    Parameters
    ----------
    name : string

    callable : callable object

    args : Value tuple

    Returns
    -------
    Value
    '''
    value = callable(*(arg.value for arg in args))
    distribution = pyro.deterministic(name, value)
    return Value(value, distribution)
