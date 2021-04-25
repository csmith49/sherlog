import torch
import torch.nn as nn
from .example import Image

def add(x, y): return x + y

def neural_predicate(network, image):
    dataset = str(image.functor)
    index = int(image.args[0])
    image = Image(dataset, index).vector()
    return network.net(image.unsqueeze(0)).squeeze(0)

class MNISTModule(nn.Module):
    def __init__(self):
        super(MNISTModule, self).__init__()
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

    def forward(self, x):
        x = self.encoder(x).view(-1, 16 * 4 * 4)
        return self.classifier(x)