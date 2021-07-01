"""Sherlog evaluation follows a functor design pattern."""

from .value import Value, Identifier, Literal
from .store import Store
from .assignment import Assignment

from typing import Generic, TypeVar, Callable, List, Dict

T = TypeVar('T')

class Functor(Generic[T]):
    """Functor with target domain `T`."""

    def __init__(
        self,
        wrap : Callable[..., T],
        fmap : Callable[[Callable, List], T],
        builtins : Dict[str, Callable[..., T]]
    ):
        """Construct a functor from `wrap` and `fmap`.

        Parameters
        ----------
        wrap : Callable[..., T]

        fmap : Callable[Callable, List], T]

        builtins : Dict[str, Callable[..., T]]
        """
        self.wrap, self.fmap, self.builtins = wrap, fmap, builtins

    def evaluate(self, obj, store : Store, **kwargs) -> T:
        """Evaluate an object in the context of the provided store.
        
        Parameters
        ----------
        obj : Any
        
        store : Store
        
        **kwargs
            Passed to `self.wrap` on execution
            
        Returns
        -------
        T
        """
        if isinstance(obj, Identifier):
            print(store)
            return store[obj]
        elif isinstance(obj, Literal):
            return self.wrap(obj.value, **kwargs)
        else:
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
            callable = store.lookup_callable(assignment.guard)
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
