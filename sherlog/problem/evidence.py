class Evidence:
    def __init__(self, atoms, bindings=None, source=None):
        self.atoms = atoms
        self.bindings = bindings
        self.source = source

    @property
    def is_parameterized(self):
        return (self.bindings is not None) and (self.source is not None)

    @classmethod
    def of_json(cls, json):
        atoms = json["atoms"]
        try:
            bindings = json["bindings"]
            source = json["source"]
        except KeyError:
            bindings, source = None, None
        return cls(atoms, bindings=bindings, source=source)

    def concretize(self, namespace):
        '''Returns an iterable over all concretized (non-parameterized) pieces of evidence.

        Parameters
        ----------
        namespace : Namespace

        Returns
        -------
        Dictionary mapping names to external objects
        '''
        if self.is_parameterized: # construct the map for each item in the dataset
            dataset = namespace[self.source]
            for obj in dataset:
                yield {k : v for k, v in zip(self.bindings, obj)}
            pass
        else: yield {} # otherwise, just send a blank map