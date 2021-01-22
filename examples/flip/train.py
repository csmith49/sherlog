import sherlog
import click
from hashids import Hashids
from random import randint
from torch.optim import SGD, Adam
from alive_progress import alive_bar
from time import sleep
from sherlog.instrumentation import Instrumenter
import storch

inst = Instrumenter("likelihood.jsonl", context={"problem" : "flip", "method" : "sgd"})
hashids = Hashids()

# delay to let the server spin up
sleep(1)

# load the problem
problem = sherlog.problem.load("./flip.sl")

@click.command()
@click.option("--epochs", default=700, help="Number of training epochs")
@click.option("--optimizer", default="sgd",
    type=click.Choice(['sgd', 'adam'], case_sensitive=False),
    help="The optimization strategy")
@click.option("--learning-rate", default=0.01, help="The learning rate of the optimizer")
@click.option("--mcmc-size", default=50, help="The number of samples used to approximate a gradient")
@click.option("--log/--no-log", default=False, help="Enables/disables recording of results")
def train(epochs, optimizer, learning_rate, mcmc_size, log):
    seed = hashids.encode(randint(0, 100000))

    optim = {
        "sgd" : SGD,
        "adam" : Adam
    }[optimizer](problem.parameters(), lr=learning_rate)

    instrumenter = Instrumenter("flip-results.jsonl", context={
        "optimizer" : optimizer,
        "learning-rate" : learning_rate,
        "mcmc-size" : mcmc_size,
        "seed" : seed
    })

    with alive_bar(epochs) as bar:
        for i in range(epochs):
            for stories in problem.stories():
                optim.zero_grad()
                loss = problem.objective(stories)
                storch.backward()
                optim.step()
                problem.clamp_parameters()
            bar()

            if i % 100 == 0:
                likelihood = problem.log_likelihood(num_samples=mcmc_size)
                instrumenter.emit(
                    likelihood=likelihood.item(),
                    step=i,
                    p=problem._parameters["p"].value.item(),
                    q=problem._parameters["q"].value.item()
                )

    # print the final parameters
    print("Learned parameters:")
    for name, param in problem._parameters.items():
        print(f"\t{name} -- {param.value:f}")

    likelihood = problem.log_likelihood(num_samples=1000)
    print(f"Final log-likelihood: {likelihood:f}")

    if log: instrumenter.flush()

if __name__ == "__main__":
    train()