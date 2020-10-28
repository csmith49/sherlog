class Context:
    def __init__(self, parameters, namespaces):
        self.store = {}
        self.parameters = parameters
        self.namespaces = namespaces

    def register(self, variable, value):
        self.store[variable] = value
    
    def lookup(self, name):
        # check the store first
        try:
            return self.store[name]
        except KeyError: pass
        # then parameters
        try:
            return self.parameters[name].value
        except KeyError: pass
        # check the namespaces
        for _, namespace in self.namespaces.items():
            try:
                return namespace.module[name]
            except KeyError: pass
        # default is to throw exception
        raise KeyError()

    def __setitem__(self, key, value):
        self.register(key, value)

    def __getitem__(self, key):
        return self.lookup(key)