from pyro import sample, param
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.distributions import Delta
from pyro.distributions import *

from torch import tensor
from torch.distributions.constraints import simplex

from random import choices

@config_enumerate
def model(data):
    weights = param("weights", tensor([1, 1, 1]) / 3, constraint=simplex)

    for trial, result in enumerate(data):
        left = sample(f"left_{trial}", Categorical(probs=weights))
        right = sample(f"right_{trial}", Categorical(probs=weights))
        total = sample(f"total_{trial}", Delta(left + right), obs=result)

def guide(data):
    pass

svi = SVI(model, guide, optim=Adam({"lr" : 0.01, "betas" : (0.9, 0.999)}), loss=TraceEnum_ELBO())

def sample_data(weights):
    trits = [-1.0, 0.0, 1.0]
    left = choices(trits, weights)[0]
    right = choices(trits, weights)[0]
    return tensor(left + right)

generated_data = [sample_data([0.3, 0.2, 0.5]) for _ in range(100)]

for epoch in range(1000):
    loss = svi.step(generated_data)
    print(f"Epoch {epoch}: loss={loss}, weights={param('weights').tolist()}") 