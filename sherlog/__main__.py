import click
from .interface import console
from .logs import enable_verbose_output
from .problem import load
from .inference import Optimizer
from torch.optim import SGD, Adam
from rich.progress import track

@click.group()
@click.option("-v", "--verbose", is_flag=True,
    envvar="SHERLOG_VERBOSE", help="Enable/disable verbose output via logging")
def main(verbose): 
    """Simple wrapper around common manipulations of Sherlog programs."""
    if verbose: enable_verbose_output()

@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-e", "--epochs", default=300, help="Number of training epochs")
@click.option("-o", "--optimizer", default="sgd",
    type=click.Choice(["sgd", "adam"], case_sensitive=False), 
    help="Torch optimizer used to train parameters")
@click.option("-l", "--learning-rate", default=0.01, help="Optimizer learning rate")
def train(filename, epochs, optimizer, learning_rate):
    """Train FILENAME with the provided parameters."""
    
    # load the problem and build the optimizer
    problem = load(filename)

    opt = {
        "sgd" : SGD,
        "adam" : Adam
    }[optimizer](problem.parameters(), lr=learning_rate)

    # construct the optimization loop
    optimizer = Optimizer(problem, opt)

    for epoch in track(range(epochs), description="Training"):
        with optimizer as o:
            ll = problem.log_likelihood(samples=1000)
            print(ll)
            o.maximize(ll)

    # print the values of the learned parameters
    for name, parameter in problem._parameters.items():
        print(f"{name}: {parameter.value:f}")

if __name__ == "__main__":
    main()