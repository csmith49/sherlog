"""Sherlog example: model selection.

TODO - instrumentation and performance.
"""

from sherlog.inference.embedding import PartitionEmbedding
import click
from torch import tensor, Tensor

from random import random, randint
from sherlog.program import loads
from sherlog.inference import minibatch, Optimizer
from sherlog.interface import console, initialize

from typing import Tuple

SOURCE = \
"""
# parameters
!parameter prior : unit.
!parameter control : unit.
!parameter treated : unit.

# facts
person(c).
person(t).

# rules
effectiveness(true; bernoulli[prior]).
effective <- effectiveness(_, 1.0).
ineffective <- effectiveness(_, 0.0).

outcome(P; bernoulli[control]) <- ineffective, person(P).
outcome(c; bernoulli[control]) <- effective, person(c).
outcome(t; bernoulli[treated]) <- effective, person(t).

control_trial(R) <- outcome(c, R).
treated_trial(R) <- outcome(t, R).
"""

@click.command()
@click.option("--effective/--not-effective", default=True, help="Is the intervention effective?")
@click.option("--control", type=float, default=0.1, help="Probability of positive outcome given no intervention.")
@click.option("--treated", type=float, default=0.8, help="Probability of positive outcome given an intervention.")
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
def cli(effective, control, treated, train, batch_size, epochs, learning_rate):
    
    # initialize!
    initialize(port=8007)

    # load the program
    program, _ = loads(SOURCE)

    # load the data
    def sample(effective, control, treated) -> Tuple[bool, bool]:
        # control
        if randint(0, 1):
            return (False, random() <= control)
        # effective treatment
        elif effective:
            return (True, random() <= treated)
        # ineffective treatment
        else:
            return (True, random() <= control)
            
    data = [sample(effective, control, treated) for _ in range(train)]
    embedder = PartitionEmbedding({
        (True, True) : "treated_trial(1.0)",
        (True, False) : "treated_trial(0.0)",
        (False, True) : "control_trial(1.0)",
        (False, False) : "control_trial(0.0)"
    })

    # build the optimizer
    optimizer = Optimizer(program, learning_rate=learning_rate)

    # iterate over the data, and optimize
    for batch in minibatch(data, batch_size, epochs=epochs):
        optimizer.maximize(*embedder.embed_all(batch.data))
        optimizer.optimize()

    # report the parameter
    for parameter in program._parameters:
        console.print(parameter.name, parameter.value)

if __name__ == "__main__":
    cli()