from .program import Program
from .evidence import Evidence

from ..interface import parse_source

from typing import Tuple, List, Optional, Mapping, Any

def loads(source : str, locals : Optional[Mapping[str, Any]] = None) -> Tuple[Program, List[Evidence]]:
    """Load a program/evidence pair from a string."""

    program, evidence = parse_source(source)
    return Program.of_json(program, locals=locals), [Evidence.of_json(evidence) for evidence in evidence]

def load(filename : str, locals : Optional[Mapping[str, Any]] = None) -> Tuple[Program, List[Evidence]]:
    """Load a program/evidence pair from file."""

    with open(filename, 'r') as f:
        contents = f.read()
    
    return loads(contents, locals=locals)