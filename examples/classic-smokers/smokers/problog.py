from .graph import Graph, Parameterization
from typing import Iterable, Optional
from itertools import chain
from subprocess import run
from math import log

PROGRAM = "/tmp/smokers-program.pl"
EVIDENCE = "/tmp/smokers-evidence.pl"

def write(filename : str, lines : Iterable[str]):
    """Write a sequence of lines to a file. Appends newlines.

    Parameters
    ----------
    filename : str
    lines : Iterable[str]
    """
    with open(filename, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")

FIT_TEMPLATE = [
    "t(_) :: stress(X) :- person(X).",
    "t(_) :: asthma_latent(X) :- person(X).",
    "t(_) :: asthma_smokes(X) :- person(X).",
    "t(_) :: influences(X, Y) :- friend(X, Y).",
    "smokes(X) :- stress(X).",
    "smokes(X) :- influences(X, Y), smokes(Y).",
    "asthma(X) :- asthma_latent(X).",
    "asthma(X) :- smokes(X), asthma_smokes(X)."   
]

def structure(graph : Graph, index : Optional[int] = None) -> Iterable[str]:
    for p in graph.people(index=index):
        yield f"person({p})."
    for p, q in graph.friends(index=index):
        yield f"friend({p}, {q})."

def evidence(graph : Graph, index : Optional[int] = None) -> Iterable[str]:
    for p in graph.smokes(True, index=index):
        yield f"evidence(smokes({p}), true)."
    for p in graph.smokes(False, index=index):
        yield f"evidence(smokes({p}), false)."
    for p in graph.asthma(True, index=index):
        yield f"evidence(asthma({p}), true)."
    for p in graph.asthma(False, index=index):
        yield f"evidence(asthma({p}), false)."

def combine(*evidences, separator="---") -> Iterable[str]:
    """Combine evidence with separators."""
    for evidence in evidences:
        yield separator
        yield from evidence

def query(graph : Graph, index : Optional[int] = None) -> Iterable[str]:
    # get the atoms for the hypothesis
    def atoms():
        for p in graph.smokes(True, index=index):
            yield f"smokes({p})"
        for p in graph.smokes(False, index=index):
            yield f"\\+smokes({p})"
        for p in graph.asthma(True, index=index):
            yield f"asthma({p})"
        for p in graph.asthma(False, index=index):
            yield f"\\+asthma({p})"
    
    yield f"q <- {', '.join(atoms())}."
    yield "query(q)."

def fit(*graphs : Graph) -> Parameterization:
    """Uses Problog to fit a set of parameters to the given graphs.

    Parameters
    ----------
    *graphs : Graph

    Returns
    -------
    Parameterization
    """
    # write the program
    graph_structure = [structure(g, index=i) for i, g in enumerate(graphs)]
    write(PROGRAM, chain(FIT_TEMPLATE, *graph_structure))

    # and write the evidence
    graph_evidence = [evidence(g, index=i) for i, g in enumerate(graphs)]
    write(EVIDENCE, combine(*graph_evidence))

    # run the program
    args = ["problog", "lfi", PROGRAM, EVIDENCE, "-k", "sdd"]
    result = run(args, capture_output=True, text=True)
    
    # parse the results
    raw_result = result.stdout.split()
    # probably worth logging
    _ = float(raw_result[0]) # log-likelihood
    _ = int(raw_result[-1]) # steps

    # extracting the parameters from the split is very manual
    parameter_values = [float(p) for p in " ".join(raw_result[1:5])[1:-1].split(", ")] # I'm sorry
    parameter_names = ["stress", "spontaneous", "comorbid", "influence"]
    parameters = Parameterization(**dict(zip(parameter_names, parameter_values)))

    return parameters

def log_likelihood(p : Parameterization, graph : Graph) -> float:
    # construct the program from p
    program = [
        f"{p.stress} :: stress(X) :- person(X).",
        f"{p.spontaneous} :: asthma_latent(X) :- person(X).",
        f"{p.comorbid} :: asthma_smokes(X) :- person(X).",
        f"{p.influence} :: influences(X, Y) :- friend(X, Y).",
        "smokes(X) :- stress(X).",
        "smokes(X) :- influences(X, Y), smokes(Y).",
        "asthma(X) :- asthma_latent(X).",
        "asthma(X) :- smokes(X), asthma_smokes(X)."   
    ]

    write(PROGRAM, chain(
        program,
        structure(graph),
        query(graph)
    ))

    # run the program
    args = ["problog", PROGRAM, "--knowledge", "sdd"]
    result = run(args, capture_output=True, text=True)

    # and capture the raw results
    log_likelihood = log(float(result.stdout.split()[-1]))
    
    return log_likelihood