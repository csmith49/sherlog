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

class Function:
    def __init__(self, symbol, arguments):
        self.symbol = symbol
        self.arguments = arguments

    def __eq__(self, other):
        try:
            (self.symbol == other.symbol) and all(l == r for l, r in zip(self.arguments, other.arguments))
        except: False

    def __str__(self):
        return f"{self.symbol}({', '.join(self.arguments)})"

# conversion
def of_json(json):
    type = json["type"]
    # variables
    if type == "variable":
        name = json["value"]
        return Variable(name)
    # functions
    elif type == "function":
        args = [of_json(arg) for arg in json["arguments"]]
        symbol = json["function"]
        return Function(symbol, args)
    # integers
    elif type == "integer":
        return json["value"]
    # booleans
    elif type == "boolean":
        return json["value"]
    # floats
    elif type == "float":
        return json["value"]
    # constants
    elif type == "constant":
        return json["value"]
    # crash
    else:
        raise NotImplementedError()