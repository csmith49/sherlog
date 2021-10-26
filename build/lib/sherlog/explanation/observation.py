from ..pipe import Value, Identifier, Literal, Statement

from typing import Mapping, Iterable

class Observation:
    """Observations pair identifiers with observed logical values."""

    def __init__(self, mapping : Mapping[str, Value]):
        """Construct an observation."""
        
        self.mapping = mapping

    # PROPERTIES

    @property
    def is_empty(self) -> bool:
        """Check if any identifiers are observed."""

        return len(self.mapping) == 0

    @property
    def domain(self) -> Iterable[Identifier]:
        """The domain of the observation."""

        for key in self.mapping.keys():
            yield Identifier(key)

    @property
    def codomain(self) -> Iterable[Value]:
        """The codomain of the observation."""

        yield from self.mapping.values()

    # STUBS

    def default_stub(self, key : str, default=None) -> Iterable[Statement]:
        """Stub for evaluating an *empty* observation."""
        
        yield Statement(f"sherlog:target:{key}", "identity", [Literal(default)])

    def check_stub(self, key : str) -> Iterable[Statement]:
        """Stub for evaluating a *non-empty* observation."""

        yield Statement(f"sherlog:keys:{key}", "tensorize", list(self.domain))
        yield Statement(f"sherlog:vals:{key}", "tensorize", list(self.codomain))
        yield Statement(f"sherlog:target:{key}", "target", [
            Identifier(f"sherlog:keys:{key}"),
            Identifier(f"sherlog:vals:{key}")
        ])

    def stub(self, key : str, default=None) -> Iterable[Statement]:
        """Stub for evaluating an observation."""

        if self.is_empty:
            yield from self.default_stub(key, default)
        else:
            yield from self.check_stub(key)

    def target(self, key : str) -> Value:
        return Identifier(f"sherlog:target:{key}")


    # MAGIC METHODS

    def __str__(self):
        return str(self.mapping)

    def __rich_repr__(self):
        yield from self.mapping.items()

    # SERIALIZATION
    
    @classmethod
    def of_json(cls, json) -> 'Observation':
        """Construct an observation from a JSON-like object."""

        if json["type"] != "observation":
            raise TypeError(f"{json} does not represent an observation.")
        
        mapping = {key : Value.of_json(value) for key, value in json["items"].items()}
        return cls(mapping)

    def to_json(self):
        """Construct a JSON-like encoding of the observation."""

        return {
            "type" : "observation",
            "items" : {
                key : value.to_json() for key, value in self.mapping.items()
            }
        }