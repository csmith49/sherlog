"""
"""

from enum import Flag
import click

from random import random
from sherlog.program import loads
from sherlog.inference import minibatch, FunctionalEmbedding, Optimizer
from sherlog.interface import print, initialize, minotaur
from torch import tensor
import torch.distributions as dist

PROG = \
"""
!parameter mu_effective : real.
!parameter sigma_effective : positive.
!parameter mu_control : real.
!parameter sigma_control : positive.
!parameter p : unit.

effective(; {no, yes} <~ bernoulli[p]).

happiness(P; normal[mu_control, sigma_control]) <- treatmentGroup(P, treatment), effective(no).
happiness(P; normal[mu_effective, sigma_effective]) <- treatmentGroup(P, treatment), effective(yes).
happiness(P ;normal[mu_control, sigma_control]) <- treatmentGroup(P, control).
"""

def model():
    p = 0.1
    data = []
    treatmentDist = dist.Normal(1., 0.5)
    controlDist = dist.Normal(0., 1.)
    obs = []
    for i in range(2):
        treatment = ["control", "treatment"][i % 2]
        data.append(f"treatmentGroup({i}, {treatment}).")
        if treatment:
            obs.append(f"happiness({i}, {treatmentDist.sample():.2f})")
        else:
            obs.append(f"happiness({i}, {controlDist.sample():.2f})")
    return data, obs

@click.command()
@click.option("-p", "--probability", type=float, default=0.6, help="Probability of a coin flip coming up heads.")
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
@click.option("-i", "--instrumentation", type=str, help="Filepath to save instrumentation logs to.")
@click.option("-s", "--samples", type=int, default=1, help="Number of samples for each explanation log-prob approximation.")
@click.option("-f", "--forcing", is_flag=True, help="Forcing explanation executions.")
@click.option("-c", "--caching", is_flag=True, help="Caching explanation sampling.")
def cli(probability, train, batch_size, epochs, learning_rate, instrumentation, samples, forcing, caching):
    """Train a simple coin flip program."""

    # initialize!
    print("Initializing...")

    initialize(port=8007, instrumentation=instrumentation)

    minotaur.enter("coin-flip")

    minotaur["probability"] = probability
    minotaur["train"] = train
    minotaur["batch-size"] = batch_size
    minotaur["epochs"] = epochs
    minotaur["learning-rate"] = learning_rate


    data, obs = model()

    # load the program
    print("Loading the program...")
    SOURCE = "\n".join(data) + PROG
    print(SOURCE)
    program, _ = loads(SOURCE)


    embedder = FunctionalEmbedding(evidence=lambda s: s)

    # build the optimizer
    print(f"Initializing the optimizer with a learning rate of {learning_rate}...")

    optimizer = Optimizer(program, learning_rate=learning_rate, samples=samples, force=forcing, cache=caching)

    old_batch_loss = tensor(0.0)

    # iterate over the data, and optimize
    for batch in minibatch(obs, batch_size, epochs=epochs):
        with minotaur("batch"):
            # we print out a "frame" for each batch for debugging purposes...
            print(f"\n🪙 Batch {batch.index:03d} in Epoch {batch.epoch:03d} 🪙")
            minotaur["batch"] = batch.index
            minotaur["epoch"] = batch.epoch

            # okay, now let's optimize
            optimizer.maximize(*embedder.embed_all(batch.data))
            batch_loss = optimizer.optimize()

            # what is the batch loss?
            print(f"Batch loss: {batch_loss:.3f} (Δ={old_batch_loss - batch_loss:.3f})")

            # and what is the program parameter doing?
            print("Parameter summary:")

            p = program.parameter("p")
            print(f"p={p.item():.3f}, ∇p={p.grad.item():.3f}, error=±{abs(p.item() - probability):.3f}")
            minotaur["p"] = p.item()
            minotaur["p-grad"] = p.grad.item()

            old_batch_loss = batch_loss

    minotaur.exit()

if __name__ == "__main__":
    cli()
