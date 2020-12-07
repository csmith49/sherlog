from .semantics import Algebra, run_factory
from torch import tensor, is_tensor
import torch.distributions as dists

def tag(value):
    if not is_tensor(value):
        return (tensor(value), None)
    else:
        return (value, None)

def untag(value): return value[0]

def normal(mean, sdev, *_, **_):
    dist = dists.Normal(mean[0], sdev[0])
    return (dist.rsample(), None)

def beta(alpha, beta, *_, **_):
    dist = dists.Beta(alpha[0], beta[0])
    return (dist.rsample(), None)

def bernoulli(prob, *_, relax=False, temperature=0.1, **_):
    if relax:
        temp = tensor(temperature)
        dist = dists.RelaxedBernoulli(temp, prob[0])
        return (dist.rsample(), None)
    else:
        dist = dists.Bernoulli(prob[0])
        value = dist.sample()
        return (value, dist.log_prob(value))

algebra = Algebra(
    tag,
    untag,
    {
        "normal" : normal,
        "beta" : beta,
        "bernoulli" : bernoulli
    }
)
run = run_factory(algebra)