from .data import Graph, Parameterization
from typing import Iterable, Optional
from itertools import chain
from subprocess import run
from math import log, exp
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

SOURCE = """
{stress} :: smokes(X) :- person(X).
{spontaneous} :: asthma(X) :- person(X).
{comorbid} :: asthma(X) :- smokes(X).
{influence} :: influence(X, Y) :- friend(X, Y).

smokes(X) :- influence(X, Y), smokes(Y).
"""

def combine(*evidences, separator="---") -> Iterable[str]:
    """Combine evidence with separators."""
    for ev in evidences:
        yield separator
        yield from ev

def evidence(graph : Graph, index : Optional[int] = None) -> Iterable[str]:
    for p in graph.smokes(True, index=index):
        yield f"evidence(smokes({p}), true)."
    for p in graph.smokes(False, index=index):
        yield f"evidence(smokes({p}), false)."
    for p in graph.asthma(True, index=index):
        yield f"evidence(asthma({p}), true)."
    for p in graph.asthma(False, index=index):
        yield f"evidence(asthma({p}), false)."

def query(graph : Graph, index : Optional[int] = None, force_target = None) -> Iterable[str]:
    # get the atoms for the hypothesis
    def atoms():
        for p in graph.smokes(True, index=index):
            yield f"smokes({p})"
        for p in graph.smokes(False, index=index):
            yield f"\\+smokes({p})"
        for p in graph.asthma(True, index=index, force_target=force_target):
            yield f"asthma({p})"
        for p in graph.asthma(False, index=index, force_target=force_target):
            yield f"\\+asthma({p})"
    
    yield f"q <- {', '.join(atoms())}."
    yield "query(q)."

def source(parameterization, train=True):
    if train:
        items = {k : f"t({v})" for k, v in parameterization.items()}
    else:
        items = dict(parameterization.items())
    return SOURCE.format(**items)

def structure(graph : Graph, index : Optional[int] = None) -> Iterable[str]:
    for p in graph.people(index=index):
        yield f"person({p})."
    for p, q in graph.friends(index=index):
        yield f"friend({p}, {q})."

class ProblogModel:
    def __init__(self):
        self._parameterization = Parameterization(0.5, 0.5, 0.5, 0.5)

    def fit(self, train):
        logger.info(f"Fitting Problog program to {len(train)} graphs.")

        # write the program
        graph_structure = [structure(g, index=i) for i, g in enumerate(train)]
        write(PROGRAM, chain([source(self._parameterization)], *graph_structure))

        # and write the evidence
        graph_evidence = [evidence(g, index=i) for i, g in enumerate(train)]
        write(EVIDENCE, combine(*graph_evidence))

        # run the program
        logger.info("Starting external Problog in lfi mode.")
        args = ["problog", "lfi", PROGRAM, EVIDENCE, "-k", "sdd"]
        result = run(args, capture_output=True, text=True)
        logger.info("External Problog run complete.")

        # parse the results
        raw_result = result.stdout.split()
        print(raw_result)
    
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
        self._parameterization = Parameterization(**dict(zip(parameter_names, parameter_values)))

        return log_likelihood / len(train), steps
    
    def log_likelihood(self, example, force_target = None):
        # construct the program from p
        write(PROGRAM, chain(
            [source(self._parameterization, train=False)],
            structure(example),
            query(example, force_target=force_target)
        ))

        # run the program
        logger.info("Starting external Problog in inference mode.")
        args = ["problog", PROGRAM, "--knowledge", "sdd"]
        result = run(args, capture_output=True, text=True)
        logger.info("External Problog run complete.")

        # make sure nothing went wrong
        if result.stderr:
            logger.warning(f"Problog run failed with output: {result.stderr}")

        # and capture the raw results
        try:
            log_likelihood = log(float(result.stdout.split()[-1]))
        except ValueError:
            log_likelihood = float("-inf")
        logger.info(f"Log-likelihood: {log_likelihood:3f}")

        return log_likelihood

    def classification_task(self, example):
        asthma = self.log_likelihood(example, force_target=True)
        not_asthma = self.log_likelihood(example, force_target=False)
        if asthma >= not_asthma:
            confidence = 1.0
        else:
            confidence = 0.0
        return confidence, example.target_classification()
