from .problem import Problem
from ..engine import parse
from .namespace import tag

def load_problem_file(filepath):
    with open(filepath, "r") as f:
        contents = f.read()
    json = parse(contents)
    return Problem.of_json(json)