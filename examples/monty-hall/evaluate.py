"""
"""

import click

from sherlog.interface import initialize, print, minotaur
from sherlog.program import loads
from sherlog.inference.embedding import parse_evidence

SOURCE = \
"""
pick(; {0, 1, 2} <~ discrete[3]).

swap(0; {1, 2} <~ discrete[2]).
swap(1; {0, 2} <~ discrete[2]).
swap(2; {0, 1} <~ discrete[2]).

strategy(swap, D) <- pick(D'), swap(D', D).
strategy(keep, D) <- pick(D).

outcome(0, win).
outcome(1, loss).
outcome(2, loss).

play(S, R) <- strategy(S, D), outcome(D, R).
"""

def sample():
    pass

@click.command()
@click.option("-e", "--explanations", type=int, default=1, help="Number of explanations for estimating marginals.")
@click.option("-s", "--samples", type=int, default=1, help="Number of per-explanation samples for estimating marginals.")
@click.option("-i", "--instrumentation", type=str, help="Instrumentation log destination.")
def cli(**kwargs):
    """Analyze the Monty-Hall problem."""

    # initialize
    print("Initializing...")
    initialize(port=8007, instrumentation=kwargs["instrumentation"])

    minotaur.enter("monty-hall")

    minotaur["explanations"] = kwargs["explanations"]
    minotaur["samples"] = kwargs["samples"]

    # load the program
    program, _ = loads(SOURCE)

    # unlike other scripts, no need for data sampling, embedding, or optimizing
    # instead, we look for performance wrt marginal likelihood computations

    # Q1: What are your chances of winning if you swap?
    evidence = parse_evidence("play(swap, win)")
    marginal = program.log_prob(evidence, **kwargs).exp().item()
    print(f"Swap win-rate: {marginal:.03f}")
    minotaur["swap"] = marginal

    # Q2: What are your chances of winning if you keep?
    evidence = parse_evidence("play(keep, win)")
    marginal = program.log_prob(evidence, **kwargs).exp().item()
    print(f"Keep win-rate: {marginal:.03f}")
    minotaur["keep"] = marginal

    minotaur.exit()

if __name__ == "__main__":
    cli()