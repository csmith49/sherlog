from ..program import Evidence

from typing import Optional, Mapping
from torch import Tensor
from dataclasses import dataclass

@dataclass
class Objective:
    """Objectives are declarative objects containing all the context needed to produce an optimization target from a program."""

    evidence : Evidence
    conditional : Optional[Evidence] = None
    parameters : Optional[Mapping[str, Tensor]] = None