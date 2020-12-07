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

        # construct callable keyword arguments
        try:
            kwargs = kwargs[f]
        except KeyError:
            kwargs = {}

        # find the callable
        f = lookup_callable(f, context, algebra)

        # return the resulting value
        return f(*args, target=target, **kwargs)
    return run
