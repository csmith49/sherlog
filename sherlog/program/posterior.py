import torch

class Posterior:
    def __init__(self, contexts=None, weights=None):
        if contexts is None:
            self._contexts = []
        else:
            self._contexts = contexts

        if weights is None:
            self._weights = torch.ones(len(self._contexts) + 3, requires_grad=True)
        else:
            self._weights = torch.tensor(weights, requires_grad=True)

    def parameters(self):
        yield self._weights

    @property
    def contexts(self):
        yield from self._contexts
    
    @property
    def weights(self):
        yield from self._weights.tolist()

    @property
    def parameterization(self):
        return self._weights