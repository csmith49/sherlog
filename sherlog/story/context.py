from torch import tensor, is_tensor

class Value:
    __slots__ = ("value", "distribution", "log_probs")
    def __init__(self, value, distribution, log_probs):
        '''Values are the intermediate representations of a run of a story.

        Parameters
        ----------
            value : object

            distribution : pyro.sample

            log_probs : (string, torch.tensor) dict
        '''
        if not is_tensor(value):
            self.value = tensor(value)
        else:
            self.value = value
        self.distribution = distribution
        self.log_probs = log_probs

    @classmethod
    def lift(cls, value):
        '''Lifts a value to a Value object.

        Parameters
        ----------
            value : object
        '''
        return cls(value, value, {})

class Context:
    def __init__(self, parameters, namespaces):
        '''Contexts maintain maps from identifiers to Value objects.

        Parameters
        ----------
            parameters : (string, Parameter) dict

            namespaces : (string, module) dict
        '''
        self.store = {}
        self.parameters = parameters
        self.namespaces = namespaces

    def register(self, variable, value):
        '''Registers a value in the context store.

        Parameters
        ----------
            variable : string

            value : Value
        '''
        self.store[variable] = value

    def __setitem__(self, variable, value):
        '''Index-assignment magic function as an interface to `register`.

        Parameters
        ----------
            variable : string

            value : value
        '''
        self.register(variable, value)

    def lookup_callable(self, name):
        '''Looks for a callable object with identifier `name` in the namespaces.

        Parameters
        ----------
            name : string

        Raises
        ------
            KeyError
        '''
        for _, namespace in self.namespaces.items():
            try:
                # check if whatever we find is actually callable
                value = getattr(namespace.module, name)
                if callable(value):
                    return value
            except AttributeError: pass
        # by default, throw a key error
        raise KeyError()

    def lookup_value(self, name):
        '''Looks for a value with identifier `name` in the context.

        Checks the context store first, then the provided parameters, then the namespaces.

        Parameters
        ----------
            name : string

        Raises
        ------
            KeyError
        '''
        # check the store
        # no need to lift, only values are ever added
        try:
            return self.store[name]
        except KeyError: pass

        # then check parameters
        # as these are kept at the program level, lift them to values
        try:
            param = self.parameters[name].value
            return Value.lift(param)
        except KeyError: pass

        # finally, iterate through the namespaces
        for _, namespace in self.namespaces.items():
            try:
                value = getattr(namespace.module, name)
                return Value.lift(value)
            except AttributeError: pass
        
        # like the rest of the lookups, the default is to raise an exception
        raise KeyError()

    def __getitem__(self, variable):
        '''Index-getter magic function for `lookup_value`.

        Parameters
        ----------
        variable : string
        '''
        return self.lookup_value(variable)