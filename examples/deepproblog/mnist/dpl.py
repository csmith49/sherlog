from .model import MNISTModule, neural_predicate
from .image import Example
from torch.optim import Adam
from typing import Iterable
from math import log
from problog.logic import Var
import dpl

SOURCE = """
nn(mnist_net, [X], Y, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, X2), digit(Y, Y2), Z is X2 + Y2.
"""

def translate_example(example : Example) -> str:
    return f"addition({example.left.atom}, {example.right.atom}, {example.total})."

class DPLModel:
    def __init__(self):
        self._module = MNISTModule()
        self._mnist_network = dpl.Network(self._module, "mnist_net", neural_predicate, optimizer=Adam(self._module.parameters(), lr=0.001))
        self._model = dpl.Model(SOURCE, [self._mnist_network])

    def _set_optimizer(self, learning_rate):
        """Resets the instance's networks' optimizer with the provided parameters.

        Parameters
        ----------
        learning_rate : float
        """
        self._mnist_network.optimizer.lr = learning_rate

    def fit(self, data : Iterable[Example], epochs : int = 1, learning_rate : float = 1.0, **kwargs):
        """
        Parameters
        ----------
        data : Iterable[Example]
        epochs : int (default=1)
        learning_rate : float (default=1.0)
        
        Returns
        -------
        float
        """
        # reset the learning rate
        self._set_optimizer(learning_rate)

        # conver the data to dpl queries
        training_data = dpl.loads('\n'.join([translate_example(ex) for ex in data]))

        # start the training
        dpl.train_model(
            self._model,
            training_data,
            epochs,
            dpl.Optimizer(self._model, 2) # what is this 2 doing here?
        )

    def log_likelihood(self, example, **kwargs):
        example = dpl.loads(translate_example(example))[0]
        args = list(example.args)
        args[-1:] = [Var(f"X_{i}") for i in range(1)]
        solution = self._model.solve(example(*args), None)
        return log(solution[example][0])

    def completion(self, example, **kwargs):
        example = dpl.loads(translate_example(example))
        return self._model.accuracy(example)[0][1]