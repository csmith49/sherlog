from abc import ABC

class Value(ABC):
    """Value abstract base class."""
    pass

class Variable(Value):
    def __init__(self, name):
        """Variable from a Sherlog program.

        Parameters
        ----------
        name : string
        """
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"<Variable: {self.name}>"

class Symbol(Value):
    def __init__(self, name):
        """Symbolic constant from a Sherlog program.

        Parameters
        ----------
        name : string
        """
        self.name = name
    
    def __eq__(self, other):
        return self.name == other.name

    def __str__(self):
        return self.name

def of_json(json):
    """Construct a Python object from a JSON representation.

    Parameters
    ----------
    json : JSON-like object

    Returns
    -------
    Python object
    """
    # variable
    if json["type"] == "variable":
        return Variable(json["value"])
    # constant
    elif json["type"] == "symbol":
        return Symbol(json["value"])
    # otherwise, just return the value
    else:
        return json["value"]