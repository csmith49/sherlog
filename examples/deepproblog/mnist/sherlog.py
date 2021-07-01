from .nn import MNISTNetwork, add

from sherlog.tooling.evaluation.model import Task, OptimizationModel
from sherlog.program import load_evidence
from sherlog import initialize

initialize()

# load the source on import
SOURCE = """
digit_probs(X; digit_nn[X]).
digit(X; categorical[P]) <- digit_probs(X, P).

add(X, Y; add[X, Y]).

addition(X, Y, Z) <- digit(X, X2), digit(Y, Y2), add(X2, Y2, Z).
"""

# program namespace
_namespace = {
    "add" : add,
    "digit_nn" : MNISTNetwork(squeeze=True)
}

# optimization task
task = Task(
    evidence=load_evidence("!evidence addition(left, right, total)."),
    target="total",
    input_map=lambda i: {
        "left" : i.left.data,
        "right" : i.right.data,
        "total" : i.total
    }
)

# embedding for samples
evidence = load_evidence("!evidence addition(left, right, total).")

# build desired model
model = OptimizationModel(SOURCE, _namespace, task)