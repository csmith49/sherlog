from pyro import sample, param, plate
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

from torch import tensor
from torch.distributions import Bernoulli
from torch.distributions.constraints import unit_interval

from random import random

def model(data):
    weight = param("weight", tensor(0.5), constraint=unit_interval)
    for trial, result in enumerate(data):
        flip = sample(f"flip_{trial}", Bernoulli(weight), obs=result)

def guide(data):
    pass

def data(weight, quantity):
    return [tensor(1.0) if random() <= weight else tensor(0.0) for _ in range(quantity)]

svi = SVI(model, guide, optim=Adam({"lr" : 0.01, "betas" : (0.9, 0.999)}), loss=Trace_ELBO())

generated_data = [tensor(1.0) if random() <= 0.7 else tensor(0.0) for _ in range(100)]

for epoch in range(1000):
    loss = svi.step(generated_data)
    print(f"Epoch {epoch}: loss={loss}, weight={param('weight').item()}") 