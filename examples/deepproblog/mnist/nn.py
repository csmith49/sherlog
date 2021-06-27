import torch
import torch.nn as nn

def add(x, y):
    return x + y

class MNISTNetwork(nn.Module):
    def __init__(self, squeeze=False):
        super(MNISTNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax(1)
        )
        self._squeeze = squeeze

    def forward(self, x):
        if self._squeeze:
            x = x.unsqueeze(0)
        x = self.encoder(x).view(-1, 16 * 4 * 4)
        result = self.classifier(x)
        if self._squeeze:
            return result.squeeze(0)
        else:
            return result