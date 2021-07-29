import sherlog
import torch.nn as nn
import torch
from .baseline import encode, decode, reconstruction_loss, kl_loss

SOURCE = """
mean(X; encode_mean_nn[X]).
latent(X; normal[M, 0.01]) <- mean(X, M).
decode(X; decode_nn[Z]) <- latent(X, Z).

kl(X; kl_loss[M, 0.01]) <- mean(X, M).
reconstruction(X; reconstruction_loss[X, Y]) <- decode(X, Y).

objective(X; add[K, R]) <- kl(X, K), reconstruction(X, R).
"""

def to_evidence(example):
    evidence_source = f"!evidence objective({example.symbol}, 0.0)."
    _, evidence = sherlog.program.loads(evidence_source)
    return evidence[0]

def to_namespace(example):
    return {
        example.symbol : example.vector.squeeze(0)
    }

def add(x, y): return x + y
def exp(x): return x.exp()

class SherlogModel:
    def __init__(self, features=16):
        self._namespace = {
            "add"                 : add,
            "encode_mean_nn"      : encode(features),
            "encode_log_sdev_nn"  : encode(features),
            "decode_nn"           : decode(features),
            "kl_loss"             : kl_loss, # TODO - check this
            "reconstruction_loss" : reconstruction_loss
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
                o.maximize(batch.objective(self._program, explanations=1, samples=1))