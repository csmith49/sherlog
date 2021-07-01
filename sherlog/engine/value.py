from abc import ABC
from typing import Any

class Value(ABC):
    """Value ABC."""
    pass

class Identifier(Value):
    """Value subclass representing Sherlog variables and symbols, i.e. things that should be given a concrete value on execution."""

    def __init__(self, name : str):
        """Construct an identifier with a given name.
        
        Parameters
        ----------
        name : str
        """
        self.name = name

    # MAGIC METHODS
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<ID: {self.name}>"

class Literal(Value):
    """Value subclass representing everything that *isn't* an identifier."""

    def __init__(self, value : Any):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"<Lit: {self.value}>"

# MONKEY PATCHING JSON CONSTRUCTOR INTO ABC

def of_json(json) -> Value:
    """Construct a Value from a JSON representation.
    
    Parameters
    ----------
    json : JSON-like object
    
    Returns
    -------
    Value
    """
    # pull the value to be wrapped
    value = json["value"]
    # check if we're an identifier
    if json["type"] in ("variable", "symbol"):
        return Identifier(value)
    else:
        return Literal(value)

Value.of_json = of_json