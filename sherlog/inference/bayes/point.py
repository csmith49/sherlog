from typing import Iterable, Tuple

class Point:
    def __init__(self, relation : str, indices : Iterable[str]):
        self.relation = relation
        self.indices = list(indices)

    @property
    def symbol(self) -> str:
        return "p:" + ":".join(self.indices)

    @property
    def key(self) -> Tuple[str]:
        return (self.relation, *self.indices)

    @property
    def evidence(self) -> str:
        return f"{self.relation}({', '.join(self.key)}, {self.symbol})"