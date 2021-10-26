from sherlog.program import loads
from sherlog.interface import initialize
from sherlog.inference import FunctionalEmbedding, Optimizer, minibatch

from random import gauss
from rich import print

# initalize the query server
initialize(port=8007)

# construct the program
SOURCE = \
"""
!parameter mu : real.
!parameter sigma : positive.
sample(S; normal[mu, sigma]).
"""

program, _ = loads(SOURCE, locals={})

# generate synthetic data
def sample(mu : float = 0.9, sigma = 0.03):
    return gauss(mu, sigma)

data = [sample() for _ in range(100)]

embedding = FunctionalEmbedding(evidence = lambda s: f"sample(seed, {s})")

# optimize
optimizer = Optimizer(program, learning_rate=1e-3)

for batch in minibatch(data, batch_size=10, epochs=10):
    optimizer.maximize(*embedding.embed_all(batch.data))
    batch_loss = optimizer.optimize()

    print(f"Batch {batch.index}:{batch.epoch} loss: {batch_loss.item()}")

# report final results
for parameter in program._parameters:
    name, value = parameter.name, parameter.value.item()
    print(f"\t{name} : {value}")