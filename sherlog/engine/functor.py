"""Sherlog evaluation follows a functor design pattern."""

from .value import Symbol, Variable

class Functor:
    def __init__(self, wrap, fmap, builtins):
        """
        Parameters
        ----------
        wrap : value -> F value

        fmap : (value -> value) -> F value -> F value

        builtins : dict[str, F value -> F value]
        """
    
        self._wrap = wrap
        self._fmap = fmap
        self._builtins = builtins

    def evaluate(self, obj, store, wrap_args={}):
        """Evaluate an object."""
        if isinstance(obj, Variable):
            return store[obj]
        elif isinstance(obj, Symbol):
            return self._wrap(store[obj], **wrap_args)
        else:
            return self._wrap(obj, **wrap_args)

    def run(self, assignment, store, wrap_args={}, fmap_args={}, parameters={}):
        # get kwargs for the function being executed
        try:
            kwargs = parameters[assignment.guard]
        except KeyError:
            kwargs = {}
        # and add the target to them (helps with, e.g., pyro)
        kwargs["target"] = assignment.target

        # evaluate arguments
        args = [self.evaluate(arg, store, wrap_args=wrap_args) for arg in assignment.arguments]

        # run the function
        try:
            callable = self._builtins[assignment.guard]
            result = callable(*args, **kwargs)
        except KeyError:
            callable = store.lookup_callable(assignment.guard)
            result = self._fmap(callable, args, kwargs, **fmap_args)
        
        store[assignment.target] = result
        return result

    def run_callable(self, callable, arguments, store, wrap_args={}, fmap_args={}):
        """Run a callable.

        Parameters
        ----------
        callable : callable

        arguments : value list

        store : Store

        wrap_args : dict option

        fmap_args : dict option

        Returns
        -------
        F value
        """
        args = [self.evaluate(arg, store, wrap_args=wrap_args) for arg in arguments]
        return self._fmap(callable, args, {}, **fmap_args)
