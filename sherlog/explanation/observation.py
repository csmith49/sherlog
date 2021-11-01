from ..pipe import Value, Identifier, Literal, Statement

from .literal import Literal, Equal
from typing import Mapping, Iterable

class Observation:
    """Observations pair identifiers with observed logical values."""

    def __init__(self, literals : Iterable[Literal]):
        """Construct an observation."""

        self.literals = list(literals)

    # PROPERTIES

    @property
    def is_empty(self) -> bool:
        """Check if the observation contains any literals."""

        return len(self.literals) == 0

    @property
    def domain(self) -> Iterable[Identifier]:
        """The domain of the observation."""

        for literal in self.literals:
            yield literal.domain

    @property
    def codomain(self) -> Iterable[Value]:
        """The codomain of the observation."""

        for literal in self.literals:
            yield literal.codomain

    @property
    def equality(self) -> Mapping[str, Value]:
        result = {
            lit.domain.value : lit.codomain for lit in self.literals if isinstance(lit, Equal)
        }
        return result

    # EVALUATION STUBS

    def target(self, key : str) -> Identifier:
        return Identifier(f"sherlog:target:{key}")

    def empty_stub(self, key : str, default = None) -> Iterable[Statement]:
        yield Statement(f"sherlog:target:{key}", "identity", [Literal(default)])

    def check_stub(self, key : str) -> Iterable[Statement]:
        targets = []
        
        for index, literal in enumerate(self.literals):
            literal_key = f"{key}:{index}"
            targets.append(literal.target(literal_key))

            yield from literal.stub(literal_key)

        yield Statement(self.target(key).value, "product", targets)

    def stub(self, key : str, default = None) -> Iterable[Statement]:
        if self.is_empty:
            yield from self.empty_stub(key, default=default)
        else:
            yield from self.check_stub(key)

    # MAGIC METHODS

    def __str__(self):
        return str(self.literals)

    def __rich_repr__(self):
        yield from self.literals

    # SERIALIZATION
    
    @classmethod
    def of_json(cls, json) -> 'Observation':
        """Construct an observation from a JSON-like object."""

        if json["type"] != "observation":
            raise TypeError(f"{json} does not represent an observation.")
        
        literals = [Literal.of_json(lit) for lit in json["literals"]]
        return cls(literals)
