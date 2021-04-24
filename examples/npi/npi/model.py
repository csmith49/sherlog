from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def fit(self, data, **kwargs): pass

    @abstractmethod
    def accuracy(self, x, y, **kwargs): pass