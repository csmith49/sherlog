from . import value

def evaluate_arguments(arguments, lift, store):
    results = []
    for argument in arguments:
        if isinstance(argument, value.Variable):
            results.append(store[argument])
        elif isinstance(argument, value.Constant):
            results.append(lift(store[argument]))
        else:
            results.append(lift(argument))
    return results

def evaluate_builtin(f, arguments, parameters={}):
    return f(*arguments, **parameters)

def evaluate_external(f, arguments, lift, unlift):
    unlifted_arguments = [unlift(arg) for arg in arguments]
    result = f(*unlifted_arguments)
    return lift(result)

def factory(lift, unlift, builtins):
    def run(statement, store, parameters={}):
        # get kwargs for the function to be executed
        try:
            kwargs = parameters[statement.function]
        except KeyError:
            kwargs = {}
        # and add the target to them (helps with, e.g., pyro)
        kwargs["target"] = statement.target

        # evaluate the arguments real quick
        arguments = evaluate_arguments(statement.arguments, lift, store)

        # run the callable
        try:
            callable = builtins[statement.function]
            result = evaluate_builtin(callable, arguments, parameters=kwargs)
        except KeyError:
            callable = store.lookup_callable(statement.function)
            result = evaluate_external(callable, arguments, lift, unlift)

        store[statement.target] = result
        return result
    return run