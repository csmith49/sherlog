from abc import ABC, abstractmethod, abstractclassmethod
from typing import Any

# ABC

class Value(ABC):
    """Abstract base class for Pipe values."""

    # IO

    @abstractclassmethod
    def of_json(cls, json):
        """Construct a value from a JSON-like object."""

        raise NotImplementedError(f"Cannot load {json}.")

    @abstractmethod
    def to_json(self):
        """Construct a JSON-like object from a value."""

        raise NotImplementedError(f"Cannot dump {self}.")

# SUBCLASSES

class Identifier(Value):
    """Pipe values that should be given a concrete value on execution."""

    def __init__(self, value : str):
        """Construct an identifier with the given name."""

        self.value = value

    # IO

    @classmethod
    def of_json(cls, json) -> "Identifier":
        """Construct an identifier from a JSON-like object."""

        if json["type"] != "identifier":
            raise TypeError(f"{json} does not represent an identifier.")
        
        value = json["value"]

        return cls(value)

    def to_json(self):
        """Construct a JSON-like object from an identifier."""

        return {
            "type" : "identifier",
            "value" : self.value
        }

    # magic methods
    
    def __str__(self):
        return self.value

class Literal(Value):
    """Pipe value with an existing interpretation."""

    def __init__(self, value : Any):
        """Construct a literal from the given value."""
        
        self.value = value

    # IO

    @classmethod
    def of_json(cls, json) -> "Literal":
        """Construct a literal from a JSON-like object."""

        if json["type"] != "literal":
            raise TypeError(f"{json} does not represent a literal.")

        value = json["value"]
        
        return cls(value)

    def to_json(self):
        """Construct a JSON-like object from a literal."""

        return {
            "type" : "literal",
            "value" : self.value
        }

    # magic methods

    def __str__(self):
        return str(self.value)

# MONKEY PATCH FOR IO ON VALUE ABC

@classmethod
def of_json(cls, json):
    """Construct a value from a JSON-like object."""

    # check if it's an identifier
    try: return Identifier.of_json(json)
    except TypeError: pass

    # then check for literal
    try: return Literal.of_json(json)
    except TypeError: pass

    # otherwise raise a type error
    raise TypeError(f"{json} does not represent a value.")

Value.of_json = of_json