"""
"""

from enum import Flag
import click

from random import random
from sherlog.program import loads
from sherlog.inference import minibatch, FunctionalEmbedding, Optimizer
from sherlog.interface import print, initialize, minotaur
import torch
from torch import tensor
import pyro.distributions as dist
import pyro
import numpy as np
from itertools import chain

NTOPICS = 2
NJUDGES = 3
JUDGES_PER_PAPER = 2
NDOCS = 5

PROG = """
meet_thresh_prob(P, J; gaussian_survival[TH, Q, E]) <- paper_quality(P, Q),
  judge_expertise(J, T, E), hasTopic(P, T), judged(J, P), judge_threshold(J, TH).
  
meets_thresh(P, J; {yes, no} <~ bernoulli[PROB]) <- meet_thresh_prob(P, J, PROB).
"""

def model():
    yesno = ["yes", "no"]

    topics = torch.randint(0, NTOPICS, (NDOCS,))

    judges = torch.tensor(np.array([
        np.random.choice(NJUDGES, JUDGES_PER_PAPER, replace=False) for _ in range(NDOCS)
    ])).long()

    with pyro.plate("judges", NJUDGES):
        threshold = pyro.sample("threshold", dist.Normal(4, 0.5))
        with pyro.plate("topics", NTOPICS):
            expertise = pyro.sample("expertise", dist.Gamma(1., 1.))
    with pyro.plate("papers", NDOCS) as papers:
        quality = pyro.sample("quality", dist.Normal(2.5, 1.0))
        with pyro.plate("paperjudge", JUDGES_PER_PAPER) as paperjudge:
            paper_expertise = torch.gather(expertise[topics, :], 0, judges.long())
            score_dist = dist.Normal(quality.float(), paper_expertise.T)
            meetsThresh = pyro.sample("meetsThresh", dist.Bernoulli(probs = 1 - score_dist.cdf(threshold[judges].T))).long()
            obs = [f"meets_thresh({p}, {judges[p,j]}, {yesno[meetsThresh[j, p]]})" for p in papers for j in paperjudge]
    
    params = [f"!parameter paperQuality{i} : real" for i in range(NDOCS)]
    params.extend([f"!parameter judgeThreshold{i} : real" for i in range(NJUDGES)])
    params.extend([f"!parameter judgeExpertise{j}_{t} : real" for j in range(NJUDGES) for t in range(NTOPICS)])
    data = list(chain.from_iterable((f"judged({j}, {p})" for (p,j) in enumerate(judges[:,i])) for i in range(JUDGES_PER_PAPER)))
    data.extend([f"hasTopic({p}, {t})" for p, t in enumerate(topics)])
    data.extend([f"judge_expertise({j}, {t}, judgeExpertise{j}_{t})" for j in range(NJUDGES) for t in range(NTOPICS)])
    data.extend([f"paper_quality({p}, paperQuality{p})" for p in range(NDOCS)])
    data.extend([f"judge_threshold({j}, judgeThreshold{j})" for j in range(NJUDGES)])

    return obs, params + data

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

    initialize(port=8022, instrumentation=instrumentation)

    minotaur.enter("coin-flip")

    minotaur["probability"] = probability
    minotaur["train"] = train
    minotaur["batch-size"] = batch_size
    minotaur["epochs"] = epochs
    minotaur["learning-rate"] = learning_rate


    obs, data = model()

    # load the program
    print("Loading the program...")
    SOURCE = ".\n".join(data) + "." + PROG
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

            old_batch_loss = batch_loss

    minotaur.exit()

if __name__ == "__main__":
    cli()
