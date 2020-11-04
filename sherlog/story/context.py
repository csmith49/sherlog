from torch import tensor, is_tensor

class Value:
    def __init__(self, value, distribution, log_prob=None):
        '''Values are the intermediate representations of a run of a story.

        Parameters
        ----------
        value : object

        distribution : pyro.sample

        log_prob : torch.tensor (default None)
        '''
        # lift the value to a tensor, if need be
        if not is_tensor(value):
            self.value = tensor(value)
        else:
            self.value = value
        self.distribution = distribution
        self.log_prob = log_prob

    @classmethod
    def lift(cls, value):
        '''Lifts a value to a Value object.

        Parameters
        ----------
        value : object
        '''
        return cls(value, value)

    @property
    def is_stochastic(self):
        '''True if the value has a log probability, false otherwise.

        Returns
        -------
        boolean
        '''
        return (self.log_prob is not None)

class Context:
    def __init__(self, maps=()):
        '''Contexts maintain maps from identifiers to Value objects.

        Parameters
        ----------
        maps : dict iter
        '''
        self.store = {}
        self._maps = list(maps)

    def register(self, variable, value):
        '''Registers a value in the context store.

        Parameters
        ----------
        variable : string

        value : Value
        '''
        if not isinstance(value, Value): raise TypeError()
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
        '''Looks for a callable object with identifier `name` in the provided maps.

        Parameters
        ----------
        name : string

        Raises
        ------
        KeyError
        '''
        for map in self._maps:
            try:
                # check if whatever we find is actually callable
                value = map[name]
                if callable(value):
                    return value
            except KeyError: pass
        # by default, throw a key error
        raise KeyError()

    def lookup(self, name):
        '''Looks for a value with identifier `name` in the context.

        Checks the context store first, then the provided maps.

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

        # then check the provided maps - lift when found
        for map in self._maps:
            try:
                value = map[name]
                return Value.lift(value)
            except KeyError: pass
        
        # by default, raise KeyError if we don't find what we're looking for
        raise KeyError()

    def __getitem__(self, variable):
        '''Index-getter magic function for `lookup`.

        Parameters
        ----------
        variable : string
        '''
        return self.lookup(variable)

    def clone(self):
        '''Constructs a fresh context with an empty store but the same maps.

        Returns
        -------
        Context
        '''
        return self.__class__(maps=tuple(self._maps))