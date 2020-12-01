import torch

class Variable:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        try:
            self.name == other.name
        except: False

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    def indexed(self, index):
        return f"{self.name}_{index}"

# conversion
def of_json(json):
    type = json["type"]
    # unit
    if type == "unit":
        return None
    # pairs
    elif type == "pair":
        left = of_json(json["left"])
        right = of_json(json["right"])
        return (left, right)
    # variables
    elif type == "variable":
        name = json["value"]
        return Variable(name)
    # functions
    elif type == "function":
        args = [of_json(arg) for arg in json["arguments"]]
        symbol = json["function"]
        return Function(symbol, args)
    # integers
    elif type == "integer":
        return torch.tensor(json["value"])
    # booleans
    elif type == "boolean":
        return json["value"]
    # floats
    elif type == "float":
        return torch.tensor(json["value"])
    # constants
    elif type == "constant":
        return json["value"]
    # crash - e.g., no support for wildcards
    else:
        raise NotImplementedError()