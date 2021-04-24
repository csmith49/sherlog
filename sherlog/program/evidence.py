class Evidence:
    def __init__(self, json):
        self.json = json

    @classmethod
    def of_json(cls, json):
        return cls(json)

    def __str__(self):
        atoms = []
        for atom in self.json["value"]:
            rel = atom["relation"]
            terms = [term["value"] for term in atom["terms"]]
            atoms.append(f"{rel}({', '.join(terms)})")
        return ", ".join(atoms)