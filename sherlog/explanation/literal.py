from ..pipe import Identifier, Value, Statement
from typing import Iterable
from abc import ABC, abstractmethod

# LITERAL ABC FOR SUBTYPING ABOVE ENUMERATED OPTIONS

class Literal(ABC):
    
    def __init__(self, variable : str, value : Value):
        """Construct a literal relating `variable` with `value`."""
        
        self.variable = variable
        self.value = value

    # PROPERTIES

    @property
    def domain(self) -> Identifier:
        """Variable being constrained by the literal."""
        
        return Identifier(self.variable)

    @property
    def codomain(self) -> Value:
        """Value constraining the variable in the literal."""
        
        return self.value

    # UTILITY/MAGIC METHODS

    def target(self, key : str) -> Identifier:
        """Identifier for storing the evaluation of the literal."""

        return Identifier(f"sherlog:literal:{key}")

    def __eq__(self, other) -> bool:
        """Equality of literals."""

        return isinstance(other, Literal) and \
                self.variable == other.variable and \
                self.value == other.value and \
                self.__class__.__name__ == other.__class__.__name__

    # ABSTRACT METHODS

    @abstractmethod
    def stub(self, key : str) -> Iterable[Statement]:
        """Evaluation stub for the literal."""

        raise NotImplementedError

# USABLE LITERAL SUBCLASSES

class Equal(Literal):
    """Represents an equality constraint `variable == target`."""

    # IMPLEMENTING ABSTRACT METHODS

    def stub(self, key : str) -> Iterable[Statement]:
        yield Statement(self.target(key).value, "semiring:one", [self.domain, self.codomain])

    # IO

    @classmethod
    def of_json(cls, json) -> "Equal":
        """Construct an equality literal from a JSON-like object."""

        if json["type"] != "equal":
            raise TypeError(f"JSON does not represent an equality literal. [json={json}]")
        
        return cls(json["variable"], Value.of_json(json["value"]))

    # MAGIC METHODS

    def __str__(self) -> str:
        return f"{self.variable} == {self.value}"

class NotEqual(Literal):
    """Represents an inequality constraint `variable != target`."""

    def stub(self, key : str) -> Iterable[Statement]:
        yield Statement(self.target(key).value, "semiring:zero", [self.domain, self.codomain])

    @classmethod
    def of_json(cls, json) -> "NotEqual":
        """Construct an inequality literal from a JSON-like object."""

        if json["type"] != "not equal":
            raise TypeError(f"JSON does not represent an inequality literal. [json={json}]")

    # MAGIC METHODS

    def __str__(self) -> str:
        return f"{self.variable} != {self.value}"

# MONKEY PATCH FOR IO ON LITERAL ABC

@classmethod
def of_json(cls, json):
    if json["type"] == "equal":
        return Equal.of_json(json)
    elif json["type"] == "not equal":
        return NotEqual.of_json(json)
    else:
        raise TypeError(f"JSON does not represent a literal. [json={json}]")

Literal.of_json = of_json
