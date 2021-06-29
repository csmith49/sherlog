from .nn import MNISTNetwork, add

from sherlog.tooling.evaluation.model import SingletonModel
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

# embedding for samples
evidence = load_evidence("!evidence addition(left, right, total).")

def namespace_generator(sample):
    return {
        "left" : sample.left.data,
        "right" : sample.right.data,
        "total" : sample.total
    }

# and build the desired model
model = SingletonModel(SOURCE, _namespace, evidence, namespace_generator)