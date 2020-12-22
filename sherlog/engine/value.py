class Variable:
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

class Constant:
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
    # unit
    if json["type"] == "unit": return None
    # pair
    elif json["type"] == "pair":
        left = of_json(json["left"])
        right = of_json(json["right"])
        return (left, right)
    # variable
    elif json["type"] == "variable":
        return Variable(json["value"])
    # constant
    elif json["type"] == "constant":
        return Constant(json["value"])
    # otherwise, just return the value
    else:
        return json["value"]