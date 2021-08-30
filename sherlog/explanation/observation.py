from ..logs import get

logger = get("explanation.observation")

from ..pipe import Value, Identifier, Literal, Statement

from typing import Mapping, Iterable, List

class Observation:
    """Observations pair identifiers with observed logical values."""

    def __init__(self, mapping : Mapping[str, Value]):
        """Construct an observation."""
        
        self.mapping = mapping

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

    def stub(self, default=None) -> List[Statement]:
        if self.is_empty:
            return [
                Statement("sherlog:target", "identity", [Literal(default)])
            ]
        else:
            return [
                Statement("sherlog:keys", "tensorize", self.domain),
                Statement("sherlog:vals", "tensorize", self.codomain),
                Statement("sherlog:goal", "target", [Identifier("sherlog:keys"), Identifier("sherlog:vals")])
            ]

    # SERIALIZATION
    
    @classmethod
    def load(cls, json) -> 'Observation':
        """Construct an observation from a JSON-like object."""

        if json["type"] == "observation":
            raise TypeError(f"{json} does not represent an observation.")
        
        mapping = {key : Value.load(value) for key, value in json["items"].items()}
        return cls(mapping)

    def dump(self):
        """Construct a JSON-like encoding of the observation."""

        return {
            "type" : "observation",
            "items" : {
                key : value.dump() for key, value in self.mapping.items()
            }
        }