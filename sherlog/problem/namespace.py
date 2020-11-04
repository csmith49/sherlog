from importlib import import_module
from inspect import getmembers

class TagError(Exception):
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return self.msg

class Register:
    def __init__(self):
        self._store = {}

    def tag(self, obj, name=None):
        if name is None:
            try: name = obj.__name__
            except AttributeError:
                raise TagError(f"Can't derive name for tagged object {obj}")
        self._store[name] = obj
        return obj

    def __call__(self, *args, **kwargs):
        return self.tag(*args, **kwargs)

    def items(self):
        yield from self._store.items()

def is_register(obj): return isinstance(obj, Register)

# the namespace class holds it all together
class Namespace:
    def __init__(self, modules=()):
        self._namespace = {}
        for module_name in modules:
            module = import_module(module_name)
            for _, register in getmembers(module, is_register):
                for name, obj in register.items():
                    self._namespace[name] = obj

    def lookup(self, key):
        return self._namespace[key]

    def __getitem__(self, key):
        return self.lookup(key)

    def items(self):
        yield from self._namespace.items()