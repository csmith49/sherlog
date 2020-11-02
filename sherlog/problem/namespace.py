from importlib import import_module
from inspect import getmembers

# useful annotating functions
def tag(obj, name=None):
    if name is None: name = obj.__name__
    setattr(obj, "sherlog:tagged", ())
    setattr(obj, "sherlog:name", name)
    return obj

def name(obj): return getattr(obj, "sherlog:name")

def is_tagged(obj): return hasattr(obj, "sherlog:tagged")

# the namespace class holds it all together
class Namespace:
    def __init__(self, modules=()):
        self._namespace = {}
        for module_name in modules:
            module = import_module(module_name)
            for obj in getmembers(module, is_tagged):
                self._namespace[name(obj)] = obj

    def lookup(self, key):
        return self._namespace[key]

    def __getitem__(self, key):
        return self.lookup(key)

    def items(self):
        yield from self._namespace.items()