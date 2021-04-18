from .graph import Graph, Parameterization
from typing import Iterable, Optional
from itertools import chain
from subprocess import run
from math import log
from sherlog.logs import get_external

logger = get_external("smokers.problog")

PROGRAM = "/tmp/smokers-program.pl"
EVIDENCE = "/tmp/smokers-evidence.pl"

def write(filename : str, lines : Iterable[str]):
    """Write a sequence of lines to a file. Appends newlines.

    Parameters
    ----------
    filename : str
    lines : Iterable[str]
    """
    logger.info(f"Writing lines to file {filename}.")
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

def program(p : Parameterization) -> Iterable[str]:
    yield from [
        f"{p.stress} :: stress(X) :- person(X).",
        f"{p.spontaneous} :: asthma_latent(X) :- person(X).",
        f"{p.comorbid} :: asthma_smokes(X) :- person(X).",
        f"{p.influence} :: influences(X, Y) :- friend(X, Y).",
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

def asthma_conditional(graph : Graph, index : Optional[int] = None) -> Iterable[str]:
    for p in graph.smokes(True, index=index):
        yield f"smokes({p})."
    for p in graph.smokes(False, index=index):
        yield f"\\+smokes({p})."
    for p in graph.asthma(True, index=index, avoid_classification_target=True):
        yield f"asthma({p})."
    for p in graph.asthma(False, index=index, avoid_classification_target=True):
        yield f"\\+asthma({p})."

    yield f"q <- asthma({graph.classification_target_symbol(index=index)})."
    yield "query(q)."

def smoking_conditional(graph : Graph, index : Optional[int] = None) -> Iterable[str]:
    for p in graph.smokes(True, index=index, avoid_classification_target=True):
        yield f"smokes({p})."
    for p in graph.smokes(False, index=index, avoid_classification_target=True):
        yield f"\\+smokes({p})."
    for p in graph.asthma(True, index=index):
        yield f"asthma({p})."
    for p in graph.asthma(False, index=index):
        yield f"\\+asthma({p})."

    yield f"q <- smokes({graph.classification_target_symbol(index=index)})."
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

    logger.info(f"Fitting Problog program to {len(graphs)} graphs.")

    # write the program
    graph_structure = [structure(g, index=i) for i, g in enumerate(graphs)]
    write(PROGRAM, chain(FIT_TEMPLATE, *graph_structure))

    # and write the evidence
    graph_evidence = [evidence(g, index=i) for i, g in enumerate(graphs)]
    write(EVIDENCE, combine(*graph_evidence))

    # run the program
    logger.info(f"Starting external Problog in lfi mode.")
    args = ["problog", "lfi", PROGRAM, EVIDENCE, "-k", "sdd"]
    result = run(args, capture_output=True, text=True)
    logger.info(f"External Problog run complete.")

    # parse the results
    raw_result = result.stdout.split()
    
    # make sure nothing went wrong
    if result.stderr:
        logger.warning(f"Problog run failed with output: {result.stderr}")

    # probably worth logging
    log_likelihood = float(raw_result[-10]) # log-likelihood
    steps = int(raw_result[-1]) # steps

    logger.info(f"Training set log-likelihood of {log_likelihood:3f} achieved with {steps} EM steps.")

    # extracting the parameters from the split is very manual
    parameter_values = [float(p) for p in " ".join(raw_result[-9:-5])[1:-1].split(", ")] # I'm sorry
    parameter_names = ["stress", "spontaneous", "comorbid", "influence"]
    parameters = Parameterization(**dict(zip(parameter_names, parameter_values)))
    logger.info(f"Resulting parameterization: {parameters}")

    return parameters

def log_likelihood(p : Parameterization, graph : Graph) -> float:
    """Compute log-likelihood of provided observation.

    Parameters
    ----------
    p : Parameterization
    graph : Graph

    Returns
    -------
    float
    """
    logger.info(f"Evaluating likelihood of {graph} with parameters {p}.")

    # construct the program from p
    write(PROGRAM, chain(
        program(p),
        structure(graph),
        query(graph)
    ))

    # run the program
    logger.info(f"Starting external Problog in inference mode.")
    args = ["problog", PROGRAM, "--knowledge", "sdd"]
    result = run(args, capture_output=True, text=True)
    logger.info(f"External Problog run complete.")

    # make sure nothing went wrong
    if result.stderr:
        logger.warning(f"Problog run failed with output: {result.stderr}")

    # and capture the raw results
    log_likelihood = log(float(result.stdout.split()[-1]))
    logger.info(f"Log-likelihood: {log_likelihood:3f}")

    return log_likelihood

def classify_asthma(p : Parameterization, graph : Graph):
    """Compute confidence that the classification target in the provided graph has asthma.

    Parameters
    ----------
    p : Parameterization
    graph : Graph

    Returns
    -------
    ???
    """
    logger.info(f"Evaluating asthma classification confidence of {graph} with parameters {p}.")

    # construct the program from p
    write(PROGRAM, chain(
        program(p),
        structure(graph),
        asthma_conditional(graph)
    ))

    # run the program
    logger.info(f"Starting external Problog in inference mode.")
    args = ["problog", PROGRAM, "--knowledge", "sdd"]
    result = run(args, capture_output=True, text=True)

    print(result)

    # make sure nothing went wrong
    if result.stderr:
        logger.warning(f"Problog run failed with output: {result.stderr}")

    # capture raw results
    confidence = float(result.stdout.split()[-1])
    
    # and get gt
    if graph.classification_target_asthma(): gt = 1.0
    else: gt = 0.0

    logger.info(f"Classification/GT: {confidence}/{gt}")

    return (confidence, gt)

def classify_smoking(p : Parameterization, graph : Graph):
    """Compute confidence that the classification target in the provided graph smokes.

    Parameters
    ----------
    p : Parameterization
    graph : Graph

    Returns
    -------
    ???
    """
    logger.info(f"Evaluating smoking classification confidence of {graph} with parameters {p}.")

    # construct the program from p
    write(PROGRAM, chain(
        program(p),
        structure(graph),
        smoking_conditional(graph)
    ))

    # run the program
    logger.info(f"Starting external Problog in inference mode.")
    args = ["problog", PROGRAM, "--knowledge", "sdd"]
    result = run(args, capture_output=True, text=True)

    # make sure nothing went wrong
    if result.stderr:
        logger.warning(f"Problog run failed with output: {result.stderr}")

    # capture raw results
    confidence = float(result.stdout.split()[-1])
    
    # and get gt
    if graph.classification_target_asthma(): gt = 1.0
    else: gt = 0.0

    logger.info(f"Classification/GT: {confidence}/{gt}")

    return (confidence, gt)