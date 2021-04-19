from .graph import Graph, Parameterization
from typing import Iterable, Optional
from itertools import chain
from math import log
from sherlog.logs import get_external
from sherlog.problem import loads
from sherlog.inference import Optimizer
from torch.optim import SGD

logger = get_external("smokers.sherlog")

FIT_TEMPLATE = [
    # parameters
    "!parameter stress : unit.",
    "!parameter spontaneous : unit.",
    "!parameter comorbid : unit.",
    "!parameter influence : unit.",
    # probabilistic rules
    "stress :: stress(X) <- person(X).",
    "spontaneous :: asthma_latent(X) <- person(X).",
    "comorbid :: asthma_smokes(X) <- person(X).",
    "influence :: influence(X, Y) <- friend(X, Y).",
    # logical rules
    "smokes(X) <- stress(X).",
    "smokes(X) <- influences(X, Y), smokes(Y).",
    "asthma(X) <- asthma_latent(X).",
    "asthma(X) <- smokes(X), asthma_smokes(X).",
    # ontological rules
    "!dependency smokes(X) | not_smokes(X) <- person(X).",
    "!dependency asthma(X) | not_asthma(X) <- person(X).",
    "!constraint smokes(X), not_smokes(X).",
    "!constraint asthma(X), not_asthma(X)."
]

def marginal_program(p : Parameterization) -> Iterable[str]:
    yield from [
        # probabilistic rules
        f"{p.stress} :: stress(X) <- person(X).",
        f"{p.spontaneous} :: asthma_latent(X) <- person(X).",
        f"{p.comorbid} :: asthma_smokes(X) <- person(X).",
        f"{p.influence} :: influence(X, Y) <- friend(X, Y).",
        # logical rules
        "smokes(X) <- stress(X).",
        "smokes(X) <- influences(X, Y), smokes(Y).",
        "asthma(X) <- asthma_latent(X).",
        "asthma(X) <- smokes(X), asthma_smokes(X).",
        # ontological rules
        "!dependency smokes(X) | not_smokes(X) <- person(X).",
        "!dependency asthma(X) | not_asthma(X) <- person(X).",
        "!constraint smokes(X), not_smokes(X).",
        "!constraint asthma(X), not_asthma(X)."
    ]

def structure(graph : Graph, index : Optional[int] = None) -> Iterable[str]:
    for p in graph.people(index=index):
        yield f"person({p})."
    for p, q in graph.friends(index=index):
        yield f"friend({p}, {q})."

def evidence(graph : Graph, index : Optional[int] = None) -> Iterable[str]:
    def atoms():
        for p in graph.smokes(True, index=index):
            yield f"smokes({p})"
        for p in graph.smokes(False, index=index):
            yield f"not_smokes({p})"
        for p in graph.asthma(True, index=index):
            yield f"asthma({p})"
        for p in graph.asthma(False, index=index):
            yield f"not_asthma({p})"
    yield f"!evidence {', '.join(atoms())}."

def fit(*graphs : Graph, learning_rate : float = 0.01, epochs : int = 10, samples : int = 10) -> Parameterization:
    """Uses Sherlog to fit a set of parameters to the given graphs.

    Parameters
    ----------
    *graphs : Graph
    learning_rate : float (default=0.01)
    epochs : int (default=10)
    samples : int (default=10)

    Returns
    -------
    Parameterization
    """

    logger.info(f"Fitting Sherlog program to {len(graphs)} graphs.")

    # build the program
    graph_structure = [structure(g, index=i) for i, g in enumerate(graphs)]
    graph_evidence = [evidence(g, index=i) for i, g in enumerate(graphs)]

    program_source = '\n'.join(chain(
        FIT_TEMPLATE,
        *chain(graph_structure),
        *chain(graph_evidence)
    ))
    
    logger.info("Parsing the program.")
    program = loads(program_source)

    # run the program
    optimizer = Optimizer(program, SGD(program.parameters(), learning_rate))

    logger.info(f"Starting training with: lr={learning_rate}, epochs={epochs}, samples={samples}")
    for epoch in range(epochs):
        logger.info(f"Learning epoch {epoch} / {epochs}.")
        with optimizer as o:
            for obj in program.objectives(epoch=epoch, samples=samples):
                o.maximize(obj)

    # and extract the parameters
    parameterization = Parameterization(**{k : v.item() for k, v in program.parameter_map.items()})
    logger.info(f"Resulting parameterization: {parameterization}")

    return parameterization

def log_likelihood(p : Parameterization, graph : Graph, stories : int = 10, samples : int = 100) -> float:
    """Compute log-likelihood of provided observation.

    Parameters
    ----------
    p : Parameterization
    graph : Graph

    stories : int (default=10)
    samples : int (default=100)

    Returns
    -------
    float
    """

    logger.info(f"Evaluating likelihood of {graph} with parameters {p}.")
    
    # build the program
    program_source = '\n'.join(chain(
        marginal_program(p),
        structure(graph),
        evidence(graph)
    ))

    logger.info("Parsing the program.")
    program = loads(program_source)

    # evaluate the log-likelihood
    logger.info(f"Computing the marginal with {stories} stories and {samples} samples.")
    log_likelihood = program.log_likelihood(stories=stories, samples=samples).item()
    logger.info(f"Log-likelihood: {log_likelihood:3f}")
    
    return log_likelihood

