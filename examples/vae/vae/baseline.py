import torch
import torch.nn as nn
from torch.distributions import Normal

def encode(features):
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, features)
    )

def decode(features):
    return nn.Sequential(
        nn.Linear(features, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Sigmoid()
    )

def reconstruction_loss(x, y):
    return nn.BCELoss(reduction='sum')(y, x)

def kl_loss(mean, log_sdev):
    log_variance = log_sdev.exp().pow(2).log()
    return -0.5 * torch.sum(1 + log_variance + mean.pow(2) - log_variance.exp())

class TorchModel:
    def __init__(self, features=16):
        self._enc_mean = encode(features)
        self._enc_sdev = encode(features)
        self._dec      = decode(features)

    def _forward(self, x):
        mean, sdev = self._enc_mean(x), self._enc_sdev(x)
        latent = Normal(mean, sdev).rsample()
        y = self._dec(latent)
        return mean, sdev, y

    def _loss(self, x, y, mean, sdev):
        return reconstruction_loss(x, y) + kl_loss(mean, sdev)

    def _parameters(self):
        yield from self._enc_mean.parameters()
        yield from self._enc_sdev.parameters()
        yield from self._dec.parameters()

    def fit(self, data, epochs : int = 1, learning_rate : int = 1):
        optimizer = torch.optim.Adam(self._parameters(), lr=learning_rate)

        for example in data:
            x = example.vector.squeeze(0)
            optimizer.zero_grad()
            mean, sdev, y = self._forward(x)
            loss = self._loss(x, y, mean, sdev)
            loss.backward()
            optimizer.step()