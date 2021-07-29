from .nn import MNISTNetwork

from torch.optim import Adam
from math import log

from sherlog.tooling.evaluation import Model
from problog.logic import Var
import dpl

# on import, load the source
SOURCE = """
nn(mnist_net, [X], Y, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) :: digit(X, Y).

number([], R, R).
number([H|T], A, R) :- digit(H, N), A2 is N + 10 * A, number(T, A2, R).
number(X, Y) :- number(X, 0, Y).

multi_addition(X, Y, Z) :- number(X, X2), number(Y, Y2), Z is X2 + Y2.
"""

# neural predicate ties the actual nn to the logic program
def neural_predicate_factory(datasource):
    # need to construct a closure for datasource
    def neural_predicate(network, term):
        # extract the Image object associated with term from the datasource

        # TODO - terms just seem to be strings at this point???
        train = str(term.functor) == "train"
        index = int(term.args[0])
        image = datasource.get(index, train=train)

        # pass the resulting image's data through the network
        return network.net(image.data.unsqueeze(0)).squeeze(0)
    # no need to wrap
    return neural_predicate

# convert sample to goal atoms
def translate(sample):
    left = [l.atom() for l in sample.left]
    right = [r.atom() for r in sample.right]
    return f"multi_addition({left}, {right}, {sample.total})."

# DPL model construction
class DPLModel(Model):
    def __init__(self, mnist_datasource):
        self._nn = MNISTNetwork()
        self._network = dpl.Network(
            self._nn,
            "mnist_net",
            neural_predicate_factory(mnist_datasource),
            optimizer=Adam(self._nn.parameters(), lr=0.001)
        )
        self._model = dpl.Model(SOURCE, [self._network])

    def fit(self, samples, *args, lr=0.001, epochs=1, **kwargs):
        # overwrite optimizer values
        self._network.optimizer.lr = lr

        # convert samples to dpl queries
        data = dpl.loads('\n'.join([translate(sample) for sample in samples]))

        # and train
        dpl.train_model(
            self._model,
            data,
            epochs,
            dpl.Optimizer(self._model, 2) # what is this 2 doing?
        )

    def log_prob(self, sample, *args, **kwargs):
        # convert sample
        ex = dpl.loads(translate(sample))[0]
        # solve the resulting problem
        args = list(ex.args)
        args[-1:] = [Var(f"X_{i}") for i in range(1)]
        solution = self._model.solve(ex(*args), None)
        # and extract the result
        return log(solution[ex][0])