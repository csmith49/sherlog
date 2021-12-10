from pyro import param, sample
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO

from torch import tensor
from torch.distributions import Normal
from torch.distributions.constraints import positive

from random import gauss

def model(data):
    mu = param("mu", tensor(0.0))
    sigma = param("sigma", tensor(1.0), constraint=positive)

    for trial, result in enumerate(data):
        draw = sample(f"sample_{trial}", Normal(mu, sigma), obs=result)

def guide(data): pass

svi = SVI(model, guide, optim=Adam({"lr" : 0.01, "betas" : (0.9, 0.999)}), loss=Trace_ELBO())

generated_data = [tensor(gauss(2.0, 0.8)) for _ in range(100)]

for epoch in range(1000):
    loss = svi.step(generated_data)
    print(f"Epoch {epoch}: loss={loss}, mu={param('mu').item()}, sigma={param('sigma').item()}") 