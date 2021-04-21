from .graph import Graph, Parameterization
from typing import Iterable, Optional
from itertools import chain
from math import log, exp
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

def evidence(graph : Graph, index : Optional[int] = None, avoid_target_smokes : bool = False, avoid_target_asthma : bool = False) -> Iterable[str]:
    def atoms():
        for p in graph.smokes(True, index=index, avoid_classification_target=avoid_target_smokes):
            yield f"smokes({p})"
        for p in graph.smokes(False, index=index, avoid_classification_target=avoid_target_smokes):
            yield f"not_smokes({p})"
        for p in graph.asthma(True, index=index, avoid_classification_target=avoid_target_asthma):
            yield f"asthma({p})"
        for p in graph.asthma(False, index=index, avoid_classification_target=avoid_target_asthma):
            yield f"not_asthma({p})"
    yield f"!evidence {', '.join(atoms())}."

def fit(*graphs : Graph,
    epochs : int = 10,
    learning_rate : float = 0.01,
    stories : int = 1,
    samples : int = 1,
    attempts : int = 100,
    seeds : int = 1,
    width : Optional[int] = None,
    depth : Optional[int] = None,
    **kwargs) -> Parameterization:
    """Uses Sherlog to fit a set of parameters to the given graphs.

    Parameters
    ----------
    *graphs : Graph
    
    **kwargs

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
            for obj in program.objectives(epoch=epoch, samples=samples, stories=stories, width=width, depth=depth, attempts=attempts, seeds=seeds):
                o.maximize(obj)

    # and extract the parameters
    parameterization = Parameterization(**{k : v.item() for k, v in program.parameter_map.items()})
    logger.info(f"Resulting parameterization: {parameterization}")

    return parameterization

def log_likelihood(p : Parameterization, graph : Graph, avoid_target_smokes : bool = False, avoid_target_asthma : bool = False,
    stories : int = 1,
    samples : int = 1,
    attempts : int = 100,
    seeds : int = 1,
    width : Optional[int] = None,
    depth : Optional[int] = None,
    **kwargs) -> float:
    """Compute log-likelihood of provided observation.

    Parameters
    ----------
    p : Parameterization
    graph : Graph

    avoid_target_smokes : bool (default=False)
    avoid_target_asthma : bool (default=False)

    Returns
    -------
    float
    """

    logger.info(f"Evaluating likelihood of {graph} with parameters {p}.")
    
    # build the program
    program_source = '\n'.join(chain(
        marginal_program(p),
        structure(graph),
        evidence(graph, avoid_target_smokes=avoid_target_smokes, avoid_target_asthma=avoid_target_asthma)
    ))

    logger.info("Parsing the program.")
    program = loads(program_source)

    # evaluate the log-likelihood
    logger.info(f"Computing the marginal with {stories} stories and {samples} samples.")
    log_likelihood = program.log_likelihood(
        stories=stories,
        samples=samples,
        attempts=attempts,
        seeds=seeds,
        width=width,
        depth=depth
    ).item()
    logger.info(f"Log-likelihood: {log_likelihood:3f}")
    
    return log_likelihood

def classify_asthma(p : Parameterization, graph : Graph, 
    stories : int = 1,
    samples : int = 1,
    attempts : int = 100,
    seeds : int = 1,
    width : Optional[int] = None,
    depth : Optional[int] = None,
    **kwargs):
    """Compute confidence thta the classification target in the provided graph has asthma.

    Sherlog has limited support for conditionals, so we use Bayes Rule to compute p(x | y) = p(x, y) / p(y).

    Parameters
    ----------
    p : Parameterization
    graph : Graph

    Returns
    -------
    Tuple[float, float]
    """

    logger.info(f"Evaluating asthma classification confidence of {graph} with parameters {p}.")

    # evaluate the joint first
    logger.info("Computing the joint.")
    joint_log_likelihood = log_likelihood(p, graph,
        stories=stories,
        samples=samples,
        attempts=attempts,
        seeds=seeds,
        width=width,
        depth=depth
    )
    logger.info(f"Joint log-likelihood: {joint_log_likelihood}")

    # then the prior
    logger.info("Computing the prior.")
    prior_log_likelihood = log_likelihood(p, graph, avoid_target_asthma=True,
        stories=stories,
        samples=samples,
        attempts=attempts,
        seeds=seeds,
        width=width,
        depth=depth
    )
    logger.info(f"Prior log-likelihood: {prior_log_likelihood}")

    # gt class confidence easily deduced
    confidence = exp(joint_log_likelihood - prior_log_likelihood)

    # get the gt and reframe confidence (if needed)
    (confidence, gt) = (confidence, 1.0) if graph.classification_target_asthma() else (1 - confidence, 0.0)
    logger.info(f"Classification confidence/gt: {confidence}/{gt}")

    return (confidence, gt)