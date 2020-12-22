from .value import Variable, Constant

class Store:
    def __init__(self, external=()):
        """A store mapping variables and constants to values.

        Parameters
        ----------
        external : iterable of mappables, default `()`
            Iterable of external maps defining the intended namespace.

        Returns
        -------
        Store
        """
        self._external = list(external)
        self._internal = {}

    def __getitem__(self, key):
        # for variables, look up in the internal store
        if isinstance(key, Variable):
            return self._internal[key.name]
        # for constants, check the external maps in order
        elif isinstance(key, Constant):
            for external_map in self._external:
                try:
                    return external_map[key.name]
                except KeyError: pass
        # otherwise, crash
        raise KeyError()

    def __setitem__(self, key, obj):
        # we can only use variables as keys - external maps handled at construction only
        if isinstance(key, Variable):
            self._internal[key.name] = obj
        else: raise ValueError()

    def lookup_callable(self, name):
        """Look up a callable object in the external maps.

        Does *not* check the existence of a `__call__` method.

        Parameters
        ----------
        name : string
            Exact name of the callable

        Returns
        -------
        callable
            Note - as of now, may not actually return a callable object

        Raises
        ------
        KeyError
        """
        for external_map in self._external:
            try:
                return external_map[name]
            except KeyError: pass
        raise KeyError()

    def __str__(self):
        return str(self._internal)