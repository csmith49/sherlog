class Evidence:
    def __init__(self, atoms, bindings=None, source=None):
        """A (possibly parameterized) piece of evidence.

        Parameters
        ----------
        atoms : JSON list

        bindings : string list, optional

        source : string, optional

        Returns
        -------
        Evidence
        """
        self.atoms = atoms
        self.bindings = bindings
        self.source = source

    @property
    def is_parameterized(self):
        """Return `True` if the evidence is parameterized.

        Returns
        -------
        boolean
        """
        return (self.bindings is not None) and (self.source is not None)

    @classmethod
    def of_json(cls, json):
        """Constructs a piece of evidence from a JSON representation.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Evidence
        """
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