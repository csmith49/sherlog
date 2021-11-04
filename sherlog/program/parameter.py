from abc import ABC, abstractmethod
from torch import Tensor, ones, tensor, softmax

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
        
        # make sure the value will accumulate gradients
        if not self.value.requires_grad:
            self.value.requires_grad = True
    
    @abstractmethod
    def clamp(self):
        """Clamp value of the parameter to the relevant domain.
        
        Modifies the object in-place.
        """
        pass

    def to_tensor(self) -> Tensor:
        return self.value

    @property
    @abstractmethod
    def domain(self) -> str:
        """Return a string representation of the parameter domain."""

        raise NotImplementedError()

    # MAGIC METHODS

    def __str__(self):
        return f"{self.value:f}"

# Utility

def initialize_parameter(default : float, dimension : int = 1):
    if dimension == 1:
        return tensor(default)
    else:
        return ones(dimension) * default

# CONCRETE PARAMETER CLASSES

class UnitIntervalParameter(Parameter):
    """Parameter restricted to the interval [0, 1]."""

    def __init__(self, name : str, dimension : int, default : float = 0.5):
        """Construct a unit parameter."""

        value = initialize_parameter(default=default, dimension=dimension)
        super().__init__(name, value)

    def clamp(self):
        """Clamp parameter to the unit interval.
        
        Modifies the parameter in-place.
        """

        self.value.clamp_(0, 1)

    @property
    def domain(self) -> str:
        """Returns a string representation of the parameter domain."""
        
        return "[0, 1]"

    def __repr__(self):
        return f"<Unit {self.name}: {self.value}>"

class PositiveRealParameter(Parameter):
    """Parameter restricted to the ray (0, infty]."""

    def __init__(self, name : str, dimension : int, default : float = 0.5):
        """Construct a positive real parameter."""

        value = initialize_parameter(default=default, dimension=dimension)
        super().__init__(name, value)
    
    def clamp(self):
        """Clamp parameter to positive ray (0, infty].
        
        Modifies the parameter in-place.
        """

        self.value.clamp_(1e-5, float("inf"))

    @property
    def domain(self):
        """Returns a string representation of the parameter domain."""

        return "ℝ⁺"

    def __repr__(self):
        return f"<Pos {self.name}: {self.value}>"

class RealParameter(Parameter):
    """Parameter on the real line."""

    def __init__(self, name : str, dimension : int, default : float = 0.5):
        """Construct a real parameter."""
        
        value = initialize_parameter(default=default, dimension=dimension)
        super().__init__(name, value)

    def clamp(self):
        """Clamp parameter to the real line.
        
        Does nothing.
        """
        pass

    @property
    def domain(self) -> str:
        """Returns a string representation of the parameter domain."""

        return "ℝ"

    def __repr__(self):
        return f"<Real {self.name}: {self.value}>"

# MONKEY PATCH

def of_json(json, default : float = 0.5) -> Parameter:
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
    name, domain, dimension = json["name"], json["domain"], json["dimension"]

    if domain == "unit":
        return UnitIntervalParameter(name, dimension, default=default)
    elif domain == "positive":
        return PositiveRealParameter(name, dimension, default=default)
    elif domain == "real":
        return RealParameter(name, dimension, default=default)
    else:
        raise NotImplementedError(f"No implementation for domain {domain}.")

Parameter.of_json = of_json
