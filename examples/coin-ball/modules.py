from torch import nn

class CoinModule(nn.Module):
    def __init__(self):
        super(CoinModule, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.classifier(x.unsqueeze(0)).float().squeeze(0)

class ColorModule(nn.Module):
    def __init__(self):
        super(ColorModule, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 3),
            nn.Softmax(1)
        )

    def forward(self, x):
        return self.classifier(x.unsqueeze(0)).float().squeeze(0)