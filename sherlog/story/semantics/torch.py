from .semantics import Algebra, run_factory
import torch.distributions as dists

def tag(value):
    if not is_tensor(value): return tensor(value)
    else: return value

def untag(value):
    return value

def normal(mean, sdev, *_, **_):
    dist = dists.Normal(mean, sdev)
    return dist.rsample()

def beta(alpha, beta, *_, **_):
    dist = dists.Beta(alpha, beta)
    return dist.rsample()

def bernoulli(prob, *_, relax=False, temperature=0.1, **_):
    if relax:
        temp = tensor(temperature)
        dist = dists.RelaxedBernoulli(temp, prob)
        return dist.rsample()
    else:
        dist = dists.Bernoulli(prob)
        return dist.sample()

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