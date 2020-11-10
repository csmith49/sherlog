import torch

m_in, m_hidden, m_out = 3, 10, 1
model = torch.nn.Sequential(
    torch.nn.Linear(m_in, m_hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(m_hidden, m_out)
)

def loss(x, y):
    return torch.dist(x, y)

day1_x = torch.tensor([0.7, 0.01, 0.4])
day1_y = torch.tensor([1.0])