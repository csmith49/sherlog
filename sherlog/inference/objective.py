from ..program import Evidence
from .bayes import Point

from typing import Iterable
from dataclasses import dataclass

@dataclass
class Objective:
    """Objectives are declarative objects containing all the context needed to produce an optimization target from a program."""

    evidence : Evidence
    points : Iterable[Point]