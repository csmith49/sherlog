from .program import Program
from .evidence import Evidence

from ..interface import parse_source

from typing import Tuple, List

def loads(source : str) -> Tuple[Program, List[Evidence]]:
    """Load a program/evidence pair from a string."""

    program, evidence = parse_source(source)
    return Program.of_json(program), [Evidence.of_json(evidence) for evidence in evidence]

def load(filename : str) -> Tuple[Program, List[Evidence]]:
    """Load a program/evidence pair from file."""

    with open(filename, 'r') as f:
        contents = f.read()
    
    return loads(contents)