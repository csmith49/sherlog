import torch
import torch.distributions.constraints as constraints

class Parameter:
    def __init__(self, name, domain, epsilon=0.001):
        '''A parameter is a tuneable symbolic constant in a SherLog program.

        Parameters
        ----------
        name : string

        domain : string

        epsilon : float (default 0.001)
        '''
        self.name = name
        self.domain = domain
        self.value = torch.tensor(0.5, requires_grad=True)
        self._epsilon = epsilon

    def constraint(self):
        '''Converts the domain of the parameter to a torch constraint.

        Returns
        -------
        torch.constraint
        '''
        if self.domain == "unit":
            return constraints.unit_interval
        elif self.domain == "positive":
            return constraints.positive
        elif self.domain == "real":
            return constraints.real
        else: raise NotImplementedError()

    @classmethod
    def of_json(cls, json):
        '''Constructs a parameter from a JSON-like object.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Parameter
        '''
        name = json["name"]
        domain = json["domain"]
        return cls(name, domain)

    def clamp(self):
        '''Clamps the value of the parameter in-place to satisfy the constraint.'''
        if self.domain == "unit":
            self.value.clamp_(0, 1)
        elif self.domain == "positive":
            self.value.clamp_(self._epsilon, float("inf"))
        else:
            pass