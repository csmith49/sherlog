"""Sherlog evaluation follows a functor design pattern."""

from .value import Value, Identifier, Literal
from .store import Store
from .assignment import Assignment

from typing import Generic, TypeVar, Callable, List, Dict
from torch import Tensor

T = TypeVar('T')

class Functor(Generic[T]):
    """Functor with target domain `T`."""

    def __init__(
        self,
        wrap : Callable[[Tensor], T],
        fmap : Callable[[Callable[..., Tensor], List[T]], T],
        builtins : Dict[str, Callable[..., T]]
    ):
        """Construct a functor from `wrap` and `fmap`.

        Parameters
        ----------
        wrap : Callable[[Tensor], T]

        fmap : Callable[Callable[..., Tensor], List[T]], T]
            Transforms functions over tensors to a function over `T`s.
            Cannot give more precise codomain, as `typing` does not support variadic signatures.

        builtins : Dict[str, Callable[..., T]]
            Builtin functions map `T`s to a `T`.
            Cannot give more precise codomain, as `typing` does not support variadic signatures.
        """
        self.wrap, self.fmap, self.builtins = wrap, fmap, builtins

    def evaluate(self, obj, store : Store[T], **kwargs) -> T:
        """Evaluate an object in the context of the provided store.
        
        Parameters
        ----------
        obj : Any
        
        store : Store[T]
        
        **kwargs
            Passed to `self.wrap` on execution.
            
        Returns
        -------
        T
        """
        # if obj is an identifier and is bound locally in the store, just return
        if isinstance(obj, Identifier) and store.is_result(obj):
            return store[obj]

        # if obj is an identifer and is a constant, return and wrap
        if isinstance(obj, Identifier) and store.is_constant(obj):
            value = store.constant(obj)
            return self.wrap(value, **kwargs)

        # if obj is a literal, unpack to tensor *then* wrap
        if isinstance(obj, Literal):
            return self.wrap(obj.to_tensor(), **kwargs)

        # otherwise, just wrap and hope for the best
        return self.wrap(obj, **kwargs)
            
    def run_assignment(self, assignment : Assignment, store : Store, **kwargs) -> T:
        """Evaluate an assignment in the context of the given store.
        
        Returns the value assigned to the target. Modifies the store in-place.
        
        Parameters
        ----------
        assignment : Assignment

        store : Store

        **kwargs
            Passed to `self.wrap` and `self.fmap` on execution.
        
        Returns
        -------
        T
        """
        # add assignment to kwargs, so functor can access naming info
        kwargs["assignment"] = assignment

        # evaluate the arguments
        args = [self.evaluate(arg, store, **kwargs) for arg in assignment.arguments]

        # builtin case
        try:
            callable = self.builtins[assignment.guard]
            result = callable(*args, **kwargs)
        # lifted case
        except KeyError:
            callable = store.callable(assignment.guard)
            result = self.fmap(callable, args, **kwargs)

        # update the store in-place
        store[assignment.target] = result
        return result
    
    def run(self, target : Identifier, guard : str, arguments : List[Value], store : Store, **kwargs) -> T:
        """Construct and evaluate an assignment.
        
        Parameters
        ----------
        target : Identifier
        
        guard : str
        
        arguments : List[Value]
        
        store : Store
        
        **kwargs
            Passed to `self.wrap` and `self.fmap` on execution.
            
        Returns
        -------
        T
        """
        assignment = Assignment(target, guard, arguments)
        return self.run_assignment(assignment, store, **kwargs)
