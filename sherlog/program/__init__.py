from .program import Program
from .evidence import Evidence
from .namespace import Register
from ..interface import parse
from typing import Tuple, Iterable

def load(filepath: str) -> Tuple[Program, Iterable[Evidence]]:
    """Load a problem from a filepath.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    Tuple[Program, Iterable[Evidence]]
    """
    with open(filepath, "r") as f:
        contents = f.read()
    return loads(contents)

def loads(contents: str) -> Tuple[Program, Iterable[Evidence]]:
    """Load a problem from a string.

    Parameters
    ----------
    contents : str

    Returns
    -------
    Tuple[Program, Iterable[Evidence]]
    """
    json = parse(contents)
    evidence = [Evidence.of_json(ev) for ev in json["evidence"]]
    problem = Program.of_json(json)
    return (problem, evidence)