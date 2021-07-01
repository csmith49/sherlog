from .value import Value, Identifier, Literal
from typing import Callable

class Store:
    def __init__(self, *builtins):
        self._builtins = list(builtins)
        self._internal = {}

    def __get_from_builtin(self, key : str):
        """Searches for an object in the builtin namespaces.

        For internal use only.

        Parameters
        ----------
        key : str

        Raises
        ------
        KeyError

        Returns
        -------
        Any
        """
        for builtins in self._builtins:
            try:
                return builtins[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __get_from_internal(self, key : str):
        """Searches for an object in the internal namespace.
        
        For internal use only.
        
        Parameters
        ----------
        key : str
        
        Raises
        ------
        KeyError
        
        Returns
        -------
        Any
        """
        return self._internal[key]

    def lookup_callable(self, name : str) -> Callable:
        """Look for a callable with the given name in the store.

        Parameters
        ----------
        name : str

        Raises
        ------
        KeyError

        Returns
        -------
        Callable
        """
        obj = self.get_from_builtin(self, name)
        if hasattr(obj, "__call__"):
            return obj
        else:
            raise KeyError(name)

    # MAGIC METHODS
    def __getitem__(self, key : Identifier):
        # try the internals
        try:
            return self.__get_from_internal(key.name)
        except KeyError:
            pass

        # try the builtins
        try:
            return self.__get_from_builtin(key.name)
        except KeyError:
            pass

        # fail
        raise KeyError(key.name)

    def __setitem__(self, key : Identifier, obj):
        self._internal[key.name] = obj

    def __str__(self):
        return str(self._internal)
