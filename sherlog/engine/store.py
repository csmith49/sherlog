from .value import Variable, Constant

class Store:
    def __init__(self, external=()):
        self._external = list(external)
        self._internal = {}

    def __getitem__(self, key):
        if isinstance(key, Variable):
            return self._internal[key.name]
        elif isinstance(key, Constant):
            for external_map in self._external:
                try:
                    return external_map[key.name]
                except KeyError: pass
        raise KeyError()

    def __setitem__(self, key, obj):
        if isinstance(key, Variable):
            self._internal[key.name] = obj
        raise ValueError()