import torch
import torch.distributions as dists

def straight_through(forward, backward):
    return forward + backward - backward.detach()

def rebar_surrogate(probs, temperature, scale):
    return scale * torch.sigmoid(probs / temperature)

def RELAX(probs, surrogate):
    # two random samples
    u_1 = dists.Uniform(0, 1).expand(probs.size()).sample()
    u_2 = dists.Uniform(0, 1).expand(probs.size()).sample()

    # u_1 defines our first continuous relaxation
    z = torch.log(probs / (1 - probs)) + torch.log(u / (1 - u))

    # discrete sample just thresholded version of z
    b = torch.heaviside(z, torch.tensor(0.0))

    # building correlated continuous relaxation
    v_0 = u_2 * (1 - probs)
    v_1 = (u_2 * probs) + (1 - probs)
    v = v_0 * (1 - b) + v_1 * b
    c = torch.log(probs / (1 - probs)) + torch.log(v / (1 - v))

    # and we use the straight-through trick to attach the gradients
    log_prob = dists.Bernoulli(probs).log_prob(b)
    backward = (b - surrogate(c)).detach() * log_prob + surrogate(z) - surrogate(c)
    return straight_through(b, backward)