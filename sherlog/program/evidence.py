from json import dumps

class Evidence:
    """Evidence is a goal conjunct."""

    def __init__(self, json):
        """Construct evidence."""
        self.json = json

    @classmethod
    def of_json(cls, json) -> 'Evidence':
        """Construct evidence from a JSON-like representation."""
        return cls(json)

    def to_json(self):
        """Construct a JSON-like encoding of the evidence."""
        return self.json

    def join(self, other):
        json = {
            "type" : "evidence",
            "value" : self.to_json()["value"] + other.to_json()["value"]
        }
        return Evidence.of_json(json)

    # MAGIC METHODS

    def __str__(self) -> str:
        atoms = []
        for atom in self.json["value"]:
            rel = atom["relation"]
            terms = [term_to_string(t) for t in atom["terms"]]
            atoms.append(f"{rel}({', '.join(terms)})")
        return ", ".join(atoms)

    def __add__(self, other) -> 'Evidence':
        return self.join(other)

def term_to_string(json):
    if json["type"] == "function":
        function_id = json["value"]
        args = ", ".join([term_to_string(arg) for arg in json["arguments"]])
        return f"{function_id}({args})"

    elif json["type"] == "unit":
        return "[]"

    else:
        return str(json["value"])