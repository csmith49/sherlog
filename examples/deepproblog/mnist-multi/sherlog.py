from .nn import MNISTNetwork, add

from sherlog.tooling.evaluation.model import Task, OptimizationModel
from sherlog.program import load_evidence
from sherlog import initialize
from torch import tensor

initialize()

# load the source on import
SOURCE = """
digit_probs(X; digit_nn[X]).
digit(X; categorical[P]) <- digit_probs(X, P).

add(X, Y; add[X, Y]).
mult(X, Y; mult[X, Y]).

concat(X, Y, Z) <- mult(10, X, B), add(B, Y, Z).

number([], X, X).
number(H :: T, A, R) <- digit(H, N), concat(A, N, A2), number(T, A2, R).

multi_addition(X, Y, Z) <- number(X, 0, X2), number(Y, 0, Y2), add(X2, Y2, Z).
"""

# evidence constructed separately
EVIDENCE = load_evidence("!evidence addition(left, right, total).")

# build desired model
model = OptimizationModel(
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