from .nn import MNISTNetwork, add

from sherlog.tooling.evaluation.model import Task, OptimizationModel, EnumerationModel
from sherlog.program import load_evidence
from sherlog import initialize
from torch import tensor

initialize()

# load the source on import
SOURCE = """
digit_probs(X; digit_nn[X] @ X).
digit(X; categorical[P] @ X) <- digit_probs(X, P).

add(X, Y; add[X, Y]).

addition(X, Y, Z) <- digit(X, X2), digit(Y, Y2), add(X2, Y2, Z).
"""

# evidence constructed separately
EVIDENCE = load_evidence("!evidence addition(left, right, total).")

# build desired model
optimization_model = OptimizationModel(
    SOURCE,
    Task(
        evidence=EVIDENCE,
        injection=lambda i: {
            "left" : i.left.data,
            "right" : i.right.data,
            "total" : tensor(i.total)
        }
    ),
    {
        "add" : add,
        "digit_nn" : MNISTNetwork(squeeze=True)
    }
)

enumeration_model = EnumerationModel(
    SOURCE,
    Task(
        evidence=EVIDENCE,
        injection=lambda i: {
            "left" : i.left.data,
            "right" : i.right.data,
            "total" : tensor(i.total)
        }
    ),
    {
        "add" : add,
        "digit_nn" : MNISTNetwork(squeeze=True)
    }
)