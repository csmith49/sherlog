from ..engine import Store

class Story:
    def __init__(self, model, observations, external=()):
        self.model = model
        self.observations = observations
        self._external = external

    @property
    def store(self):
        return Store(external=self._external)

    def run(self, algebra, parameters={}):
        store = self.store
        for stmt in self.model.statements:
            algebra(stmt, store, parameters=parameters)
        return store