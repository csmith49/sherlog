import sherlog
import torch
from .model import MNISTModule, add
from .image import Example, Image

SOURCE = """
digit_probs(X; digit_nn[X]).
digit(X; categorical[P]) <- digit_probs(X, P).

add(X, Y; add[X, Y]).

addition(X, Y, Z) <- digit(X, X2), digit(Y, Y2), add(X2, Y2, Z).
"""

def translate_example(left, right, total):
    return f"!evidence addition({left.symbol}, {right.symbol}, {total})."

def translate_examples(examples):
    for ex in examples:
        yield translate_example(ex.left, ex.right, ex.total)

def to_evidence(example):
    _, ev = sherlog.program.loads(translate_example(example.left, example.right, example.total))
    return ev[0]

def to_namespace(example):
    return {
        example.left.symbol : example.left.vector,
        example.right.symbol : example.right.vector
    }

class SherlogModel:
    def __init__(self):
        self._namespace = {
            "add" : add,
            "digit_nn" : MNISTModule(squeeze=True)
        }
        self._program, _ = sherlog.program.loads(SOURCE, namespace=self._namespace)

    def fit(self, data, epochs : int = 1, learning_rate : float = 0.01, batch_size : int = 1, **kwargs):
        optimizer = sherlog.inference.Optimizer(
            self._program,
            optimizer="adam",
            learning_rate=learning_rate
        )

        for batch in sherlog.inference.namespace_minibatch(data, batch_size=batch_size, to_evidence=to_evidence, to_namespace=to_namespace, epochs=epochs):
            with optimizer as o:
                o.maximize(batch.objective(self._program, explanations=1, samples=1000))

    def _ll(self, left, right, total, explanations, samples) -> float:
        # convert and load example
        _, evidence = sherlog.program.loads(translate_example(left, right, total))
        # register the example images in the namespace
        namespace = to_namespace(Example(left, right, total))
        # and compute!
        return self._program.log_prob(evidence[0], explanations=explanations, samples=samples, namespace=namespace).item()

    def log_likelihood(self, example, explanations : int = 1, samples : int = 100, **kwargs) -> float:
        return self._ll(example.left, example.right, example.total, explanations, samples)

    def completion(self, example, explanations : int = 1, samples : int = 100, **kwargs) -> float:
        key = lambda p: self._ll(example.left, example.right, p, explanations=explanations, samples=samples)
        mlo = max(example.output_permutations(), key=key)
        if mlo == example.total:
            return 1.0
        else:
            return 0.0