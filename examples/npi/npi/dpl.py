import dpl
import torch
from torch.optim import Adam
from typing import Iterable
from .example import Example
from .model import Model

SOURCE = """
nn(swap_net, [X,Y],Z,[no_swap, swap]) :: swap(X,Y,Z).

quicksort([X|Xs],Ys) :-
  partition(Xs,X,Left,Right),
  quicksort(Left,Ls),
  quicksort(Right,Rs),
  append(Ls,[X|Rs],Ys).
quicksort([],[]).

partition([X|Xs],Y,[X|Ls],Rs) :-
  swap(X,Y,swap), partition(Xs,Y,Ls,Rs).
partition([X|Xs],Y,Ls,[X|Rs]) :-
  swap(X,Y,no_swap), partition(Xs,Y,Ls,Rs).


partition([],Y,[],[]).

append([],Ys,Ys).
append([X|Xs],Ys,[X|Zs]) :- append(Xs,Ys,Zs).

forth_sort(L,L2) :- quicksort(L,L2).
"""

def translate_example(example : Example) -> str:
    """Convert an example to DeepProbLog syntax.

    Parameters
    ----------
    example : Example

    Returns
    -------
    string
    """
    inputs = [str(i) for i in example.input]
    outputs = [str(o) for o in example.output]
    return f"forth_sort([{', '.join(inputs)}],[{', '.join(outputs)}])."

def neural_predicate(network, i1, i2):
    d = torch.zeros(20)
    d[int(i1)]      = 1.0
    d[int(i2) + 10] = 1.0
    d = torch.autograd.Variable(d.unsqueeze(0))
    output = network.net(d)
    return output.squeeze(0)

class DPLModel(Model):
    def __init__(self):
        """DPL model for NPI testing."""
        self._fc = dpl.FC(20, 2)
        self._swap_network = dpl.Network(self._fc, "swap_net", neural_predicate, optimizer=Adam(self._fc.parameters(), lr=1.0))
        self._model = dpl.Model(SOURCE, [self._swap_network])

    def _set_optimizer(self, learning_rate):
        """Resets the instance's network's optimizer with the provided parameters.

        Parameters
        ----------
        learning_rate : float
        """
        # opt = optimizer(self._fc.parameters(), lr=learning_rate)
        self._swap_network.optimizer.lr = learning_rate

    def fit(self, data : Iterable[Example], epochs : int = 1, learning_rate : float = 1.0, **kwargs):
        """
        Parameters
        ----------
        data : Iterable[Example]
        epochs : int (default=1)
        learning_rate : float (default=0.1)

        Returns
        -------
        float
        """
        # make sure we set the optimizer before starting training (if necessary)
        self._set_optimizer(learning_rate)

        # convert the data
        training_data = dpl.loads('\n'.join([translate_example(ex) for ex in data]))

        # start the training
        dpl.train_model(
            self._model,
            training_data,
            epochs,
            dpl.Optimizer(self._model, 32) # not entire sure what the 32 controls...
        )

    def accuracy(self, example, **kwargs):
        """
        Parameters
        ----------
        example : Example

        Returns
        -------
        float
        """
        example = dpl.loads(translate_example(example))
        return self._model.accuracy(example)[0][1]