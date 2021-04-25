from .model import Model
from .example import Example
from typing import Iterable
import torch
import sherlog

SOURCE = """
swap_rate(X, Y; swap_nn[X, Y]).
swap(X, Y; categorical[R]) <- swap_rate(X, Y, R).

quicksort([], []).
quicksort(X :: XS, YS) <-
    partition(XS, X, Left, Right),
    quicksort(Left, LS),
    quicksort(Right, RS),
    append(LS, X :: RS, YS).

partition([], Y, [], []).
partition(X :: XS, Y, X :: LS, RS) <- swap(X, Y, 1), partition(XS, Y, LS, RS).
partition(X :: XS, Y, LS, X :: RS) <- swap(X, Y, 0), partition(XS, Y, LS, RS).

append([], YS, YS).
append(X :: XS, YS, X :: ZS) <- append(XS, YS, ZS).

forth_sort(I, O) <- quicksort(I, O).
"""

def translate_example(inputs, outputs) -> str:
    inputs = ", ".join([str(i) for i in inputs])
    outputs = ", ".join([str(o) for o in outputs])
    return f"!evidence forth_sort([{inputs}], [{outputs}])."

def translate_examples(examples : Iterable[Example]):
    lines = [translate_example(ex.input, ex.output) for ex in examples]
    _, evidence = sherlog.program.loads("\n".join(lines))
    return evidence

class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(20, 2),
            torch.nn.Softmax(1)
        )
    
    def forward(self, x, y):
        input_vec = torch.zeros(20)
        input_vec[int(x)] = 1.0
        input_vec[int(y) + 10] = 1.0
        output = self.nn(input_vec.unsqueeze(0))
        return output.squeeze(0)

class SherlogModel(Model):
    def __init__(self):
        self._problem, _ = sherlog.program.loads(SOURCE, namespace={"swap_nn" : Module()})

    def fit(self, data : Iterable[Example], epochs : int = 1, learning_rate : float = 1.0, batch_size : int = 1, **kwargs):
        optimizer = sherlog.inference.Optimizer(self._problem, optimizer="adam", learning_rate=learning_rate)
        
        for batch in sherlog.inference.minibatch(translate_examples(data), batch_size=batch_size, epochs=epochs):
            with optimizer as o:
                o.maximize(batch.objective(self._problem, explanations=3, samples=100, seeds=3))

    def accuracy(self, example, **kwargs):
        """Compute the 0-1 accuracy of the model on the given example.

        Returns 1.0 if the sortest list has the highest log-prob across all permutations, and 0.0 otherwise.

        Parameters
        ----------
        example : Example

        Returns
        -------
        float
        """
        # compute log-likelihood of each possible permutation
        prediction, likelihood = None, float("-inf")

        print(f"Evaluating example {example.input} / {example.output}")

        for possibility in example.output_permutations():
            _, evidence = sherlog.program.loads(translate_example(example.input, possibility))
            possibility_likelihood = self._problem.likelihood(evidence[0], explanations=5, samples=100)
            if possibility_likelihood > likelihood:
                print(f"swapping {prediction} for {possibility} w/ likelihood {possibility_likelihood}")
                prediction, likelihood = possibility, possibility_likelihood

        if list(prediction) == example.output:
            return 1.0
        else:
            return 0.0
                