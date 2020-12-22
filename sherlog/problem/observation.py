from ..engine import value, evaluate

class Observation:
    def __init__(self, mapping):
        """Observation of variable assignemnts for a story.

        Parameters
        ----------
        mapping : dict
            Each key-value pair maps a string to a value

        Returns
        -------
        Observation
        """
        self.mapping = mapping

    @classmethod
    def of_json(cls, json):
        """Build an observation from a JSON representation.

        Parameters
        ----------
        json : JSON-like object

        Returns
        -------
        Observation
        """
        mapping = {}
        for obs in json:
            mapping[obs["variable"]] = value.of_json(obs["value"])
        return cls(mapping)

    def variables(self):
        """Compute the domain of the observation.

        Returns
        -------
        Variable iterable
        """
        for k, _ in self.mapping.items():
            yield value.Variable(k)

    def evaluate(self, store, algebra):
        """Evaluate the observation.

        Parameters
        ----------
        store : Store

        algebra : Algebra

        Returns
        -------
        value
        """
        for _, v in self.mapping.items():
            yield evaluate(v, store, algebra)