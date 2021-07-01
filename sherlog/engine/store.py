from .value import Identifier
from typing import Callable, Generic, TypeVar
from torch import Tensor, tensor

from ..logs import get

logger = get("engine.store")

T = TypeVar('T')

class Store(Generic[T]):
    def __init__(self, **bindings):
        """Construct a store from a set of bindings.
        
        Parameters
        ----------
        **bindings
        """
        self._callables = {}
        self._constants = {}
        
        # sort the bindings into the above
        for key, value in bindings.items():
            # callable?
            if hasattr(value, "__call__"):
                self._callables[key] = value
            
            # tensorable?
            elif hasattr(value, "to_tensor") or isinstance(value, Tensor):
                self._constants[key] = value

            # otherwise, raise a warning and treat as a constant anyways
            else:
                logger.warning(f"Binding {key}:{value} not a tensor or callable. Tensor conversion may fail.")
                self._constants[key] = value

        # no results to begin with!
        self._results = {}

    def is_callable(self, key : str) -> bool:
        """Check if the provided key is associated with a callable.
        
        Parameters
        ----------
        key : str
        
        Returns
        -------
        bool
        """
        return key in self._callables

    def is_constant(self, key : Identifier) -> bool:
        """Check if the provided key is associated with a constant.
        
        Parameters
        ----------
        key : Identifier
        
        Returns
        -------
        bool
        """
        return key.name in self._constants

    def is_result(self, key : Identifier) -> bool:
        """Check if the provided key is associated with a result.
        
        Parameters
        ----------
        key : Identifier
        
        Returns
        -------
        bool
        """
        return key.name in self._results

    def callable(self, key : str) -> Callable[..., Tensor]:
        """Return the callable with the indicated name, if it exists.
        
        Parameters
        ----------
        key : str
        
        Raises
        ------
        KeyError
            If no such callable exists.

        Returns
        -------
        Callable[..., Tensor]
            Maps tensors to a tensor; variadic codomains not supported yet.
        """
        return self._callables[key]

    def constant(self, key : Identifier) -> Tensor:
        """Return the constant with the indicated name, if it exists.
        
        Parameters
        ----------
        key : Identifier

        Raises
        ------
        KeyError
            If no such constant exists.

        Returns
        -------
        Tensor
            Relies on `torch.tensor` and instance `to_tensor` methods to convert objects.
        """
        value = self._constants[key.name]
        
        # if value is a tensor, just return
        if isinstance(value, Tensor):
            return value

        # otherwise, check for a to_tensor instance method
        if hasattr(value, "to_tensor"):
            return value.to_tensor()

        # or cross your fingers and hope for the best
        return tensor(value)

    def result(self, key : Identifier) -> T:
        """Return the result associated with the indicated name, if it exists.
        
        Parameters
        ----------
        key : Identifier
        
        Raises
        ------
        KeyError
            If no such result exists.
            
        Returns
        -------
        T
        """
        return self._results[key.name]

    def register(self, key : Identifier, result : T):
        """Register a result in the store. Modifies the instance in-place.
        
        Parameters
        ----------
        key : Identifier

        result : T
        """
        self._results[key.name] = result

    # MAGIC METHODS
    def __getitem__(self, key : Identifier) -> T:
        return self.result(key)

    def __setitem__(self, key : Identifier, result : T):
        self.register(key, result)

    def __contains__(self, key : Identifier) -> bool:
        return key.name in self._results

    def __str__(self) -> str:
        return str(self._results)
