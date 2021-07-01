from abc import ABC, abstractmethod
from torch import Tensor, tensor

# ABSTRACT PARAMETER CLASS

class Parameter(ABC):
    """Parameters are tuneable constants in Sherlog programs."""

    def __init__(self, name : str, value : Tensor):
        """Create a parameter by name and value.
        
        Parameters
        ----------
        name : str
        
        value : Tensor
        """
        self.name, self.value = name, value
    
    @abstractmethod
    def clamp(self):
        """Clamp value of the parameter to the relevant domain.
        
        Modifies the object in-place.
        """
        pass

    def to_tensor(self) -> Tensor:
        return self.value

    # MAGIC METHODS

    def __str__(self):
        return f"<Parameter {self.name}: {self.value}>"

# CONCRETE PARAMETER CLASSES

class UnitIntervalParameter(Parameter):
    """Parameter restricted to the interval [0, 1]."""

    def __init__(self, name : str, default : float = 0.5):
        """Construct a unit parameter.
        
        Parameters
        ----------
        name : str
        
        default : float (default=0.5)
        """
        value = tensor(default)
        super().__init__(name, value)

    def clamp(self):
        """Clamp parameter to the unit interval.
        
        Modifies the parameter in-place.
        """
        self.value.clamp_(0, 1)

class PositiveRealParameter(Parameter):
    """Parameter restricted to the ray (0, infty]."""

    def __init__(self, name : str, default : float = 0.5):
        """Construct a positive real parameter.
        
        Parameters
        ----------
        name : str
        
        default : float (default=0.5)
        """
        value = tensor(default)
        super().__init__(name, value)
    
    def clamp(self):
        """Clamp parameter to positive ray (0, infty].
        
        Modifies the parameter in-place.
        """
        self.value.clamp_(self._epsilon, float("inf"))

class RealParameter(Parameter):
    """Parameter on the real line."""

    def __init__(self, name : str, default : float = 0.5):
        """Construct a real parameter.
        
        Parameters
        ----------
        name : str
        
        default : float (default=0.5)
        """
        value = tensor(default)
        super().__init__(name, value)

    def clamp(self):
        """Clamp parameter to the real line.
        
        Does nothing.
        """
        pass

# MONKEY PATCH

def of_json(json, epsilon : float = 1e-10) -> Parameter:
    """Construct a parameter from a JSON-like representation.
    
    Parameters
    ----------
    json : JSON-like object
    
    epsilon : float (default=1e-10)
        Close-to-zero value to prevent underflows.

    Raises
    ------
    NotImplementedError

    Returns
    -------
    Parameter
    """
    name, domain = json["name"], json["domain"]

    if domain == "unit":
        return UnitIntervalParameter(name)
    elif domain == "positive":
        return PositiveRealParameter(name)
    elif domain == "real":
        return RealParameter(name)
    else:
        raise NotImplementedError(f"No implementation for domain {domain}.")

Parameter.of_json = of_json
