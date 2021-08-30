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

    # MAGIC METHODS

    def __str__(self) -> str:
        atoms = []
        for atom in self.json["value"]:
            rel = atom["relation"]
            terms = [str(term["value"]) for term in atom["terms"]]
            atoms.append(f"{rel}({', '.join(terms)})")
        return ", ".join(atoms)