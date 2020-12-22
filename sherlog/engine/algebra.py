from . import value
from collections import namedtuple

Algebra = namedtuple("Algebra", "lift unlift builtins")

def evaluate(obj, store, algebra):
    if isinstance(obj, value.Variable):
        return store[obj]
    elif isinstance(obj, value.Constant):
        return algebra.lift(store[obj])
    else:
        return algebra.lift(obj)

def evaluate_arguments(arguments, store, algebra):
    results = [evaluate(arg, store, algebra) for arg in arguments]
    return results

def evaluate_builtin(f, arguments, parameters={}):
    return f(*arguments, **parameters)

def evaluate_external(f, arguments, algebra):
    unlifted_arguments = [algebra.unlift(arg) for arg in arguments]
    result = f(*unlifted_arguments)
    return algebra.lift(result)

def run(statement, store, algebra, parameters={}):
    # get kwargs for the function to be executed
    try:
        kwargs = parameters[statement.function]
    except KeyError:
        kwargs = {}
    # and add the target to them (helps with, e.g., pyro)
    kwargs["target"] = statement.target

    # evaluate the arguments real quick
    arguments = evaluate_arguments(statement.arguments, store, algebra)

    # run the callable
    try:
        callable = algebra.builtins[statement.function]
        result = evaluate_builtin(callable, arguments, parameters=kwargs)
    except KeyError:
        callable = store.lookup_callable(statement.function)
        result = evaluate_external(callable, arguments, algebra)

    store[statement.target] = result
    return result