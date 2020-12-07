from collections import namedtuple
from .. import term

# Evaluation Algebra
Algebra = namedtuple("Algebra", "tag untag builtins")

# utilities
def identity(obj): return obj

def wrap_external(callable, algebra):
    def wrapper(*args, **kwargs):
        args = (algebra.untag(a) for a in args)
        return algebra.tag(callable(*args, **kwargs))
    return wrapper

def lookup_callable(name, context, algebra):
    # check the context first
    try:
        f = context.lookup_callable(name)
        return wrap_external(f, algebra)
    except KeyError: pass

    # check builtins
    try:
        return algebra.builtins[name]
    except KeyError: pass

    # if we can't find it, oh well
    raise KeyError

def convert_arguments(arguments, context, algebra):
    args = []
    for arg in arguments:
        # if we have a variable, look it up in the context
        if isinstance(arg, term.Variable):
            value = context[arg.name] # no need to lift

        # same if we have a string - that's a symbolic value we need to concretize
        elif isinstance(arg, str):
            value = algebra.tag(context.lookup(arg))

        # otherwise, we're some object fresh from the parser and need to be lifted
        else:
            value = algebra.tag(arg)
        args.append(value)
    return args

def run_factory(algebra):
    def run(target, f, args, context, **kwargs):
        # convert the arguments
        args = convert_arguments(args, context, algebra)

        # find the callable
        f = lookup_callable(f, context, algebra)

        # construct callable keyword arguments
        try:
            kwargs = kwargs[f]
        except KeyError:
            kwargs = {}

        # return the resulting value
        return f(*args, target=target, **kwargs)

# base class



class Semantics:
    def __init__(self, maps=()):
        self._maps = list(maps)

    def tag(self, obj):
        raise NotImplementedError

    def untag(self, value):
        raise NotImplementedError

    def wrap_external_callable(self, callable):
        def wrapped(*args, **_):
            arguments = (self.untag(value) for value in args)
            return callable(*arguments)
        return wrapped

    def lookup_callable(self, name):
        """Look for a callable object with identifier `name`.

        Start by examining the provided maps (in `self.maps`). If nothing is found, look for `self._name`.

        :param string name: The name of the callable
        :return: A callable object
        :raises KeyError:
        """
        # check the maps first
        for map in self._maps:
            try:
                # check if whatever we find is actually callable
                value = map[name]
                if callable(value):
                    return self.wrap_external_callable(value)
            except KeyError: pass
        
        # see if there's a method in `self`
        try:
            return getattr(self, f"_{name}")
        except AttributeError: pass
        
        # by default, throw a key error
        raise KeyError()

    def lookup(self, name):
        """Look for a value with identifier `name`.

        Start by examining the provided maps (in `self.maps`). If nothing is found, look for `self._name`.

        :param string name: The name of the value
        :return: An object wrapped in `self.lift`
        :raises KeyError:
        """
        # check the provided maps - lift when found
        for map in self._maps:
            try:
                value = map[name]
                return self.lift(value)
            except KeyError: pass

        # then check locally
        try:
            return self.lift(getattr(self, f"_{name}"))
        except AttributeError: pass
        
        # by default, raise KeyError if we don't find what we're looking for
        raise KeyError()

    def run(self, target, function, arguments, context, **kwargs):
        """Compute `target = function(*arguments)` in `context`.
        
        :param variable target: The variable the generated value will be bound to
        :param string function: The name of the generating callable
        :param term list arguments: Arguments to the callable
        :param dict context: The execution context providing semantics for variables
        
        :return: A value as computed by the callable `function` 
        """
        # convert arguments
        args = []
        for arg in arguments:
            # if we have a variable, look it up in the context
            if isinstance(arg, term.Variable):
                value = context[arg.name] # no need to lift
            # same if we have a string - that's a symbolic value we need to concretize
            elif isinstance(arg, str):
                value = self.lookup(arg)
            # otherwise, we're some object fresh from the parser and need to be lifted
            else:
                value = self.lift(arg)
            args.append(value)

        # get callable
        f = self.lookup_callable(function)

        # get kw arguments for callable
        try:
            kwargs = kwargs[function]
        except KeyError:
            kwargs = {}

        # evaluate and return
        return f(*args, target=target, **kwargs)