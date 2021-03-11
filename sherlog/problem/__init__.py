from .problem import Problem
from .namespace import Register
from ..interface import parse

def load(filepath: str):
    """Load a problem from a filepath.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    Problem
    """
    with open(filepath, "r") as f:
        contents = f.read()
    json = parse(contents)
    return Problem.of_json(json)

def loads(contents: str):
    """Load a problem from a string.

    Parameters
    ----------
    contents : str

    Returns
    -------
    Problem
    """
    json = parse(contents)
    return Problem.of_json(json)