from . import value
from collections import namedtuple

# simple named tuple - the only structure we need is named access
Algebra = namedtuple("Algebra", "lift unlift builtins")

def evaluate(obj, store, algebra):
    """Evaluate an object.

    Parameters
    ----------
    obj : object

    store : Store

    algebra : Algebra

    Returns
    -------
    value
    """
    if isinstance(obj, value.Variable):
        return store[obj]
    elif isinstance(obj, value.Constant):
        return algebra.lift(store[obj])
    else:
        return algebra.lift(obj)

def evaluate_arguments(arguments, store, algebra):
    """Evaluate a list of arguments to a function.

    Parameters
    ----------
    arguments : object list

    store : Store

    algebra : Algebra

    Returns
    -------
    value list
    """
    results = [evaluate(arg, store, algebra) for arg in arguments]
    return results

def evaluate_builtin(f, arguments, parameters={}):
    """Evaluate a built-in function.

    Parameters
    ----------
    f : callable

    arguments : value list

    parameters : dict

    Returns
    -------
    value
    """
    return f(*arguments, **parameters)

def evaluate_external(f, arguments, algebra):
    """Evaluate an external function.

    We assume external functions operate over Python objects, not values. So we have to unlift the arguments and lift the result back into the algebra.

    Parameters
    ----------
    f : callable

    arguments : value list

    algebra : Algebra

    Returns
    -------
    value
    """
    unlifted_arguments = [algebra.unlift(arg) for arg in arguments]
    result = f(*unlifted_arguments)
    return algebra.lift(result)

def run(statement, store, algebra, parameters={}):
    """Run a statement, updating the store in-place and returning the result.

    Parameters can be passed to a built-in `f` by including a `f : kwargs` key-value pair in the optional `parameters`.

    Parameters
    ----------
    statement : Statement

    store : Store

    algebra : Algebra

    parameters : dict, optional

    Returns
    -------
    value
    """
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