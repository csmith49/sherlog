import torch

class Record:
    def __init__(self, features, context):
        self._features = features
        self._context = context

    @classmethod
    def of_json(cls, json):
        features = torch.tensor(json["features"])
        context = [torch.tensor(c) for c in json["context"]]
        return cls(features, context)

    def features(self, parameterization):
        return self._features.dot(parameterization)
    
    def context(self, parameterization):
        return [ctx.dot(parameterization) for ctx in self._context]

    def log_prob(self, parameterization):
        return self.features(parameterization).log() - torch.stack(self.context(parameterization)).sum().log()

    def __str__(self):
        return str(self._context)

    def __repr__(self):
        return str(self)

class History:
    def __init__(self, records):
        self._records = records
    
    @classmethod
    def of_json(cls, json):
        records = [Record.of_json(r) for r in json["records"]]
        return cls(records)

    def log_prob(self, parameterization):
        if self._records:
            return torch.stack([r.log_prob(parameterization) for r in self._records]).sum()
        else:
            return torch.tensor(1.0)

    def join(self, other):
        records = self._records + other._records
        return History(records)

    def __add__(self, other):
        return self.join(other)
    
    def __str__(self):
        return str(self._records)
    
    def __repr__(self):
        return str(self)