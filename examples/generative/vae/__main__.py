from . import sample, TorchModel, SherlogModel
from sherlog.logs import enable, get_external
from sherlog import console
from sherlog.tooling.instrumentation import seed, Timer
from json import dumps
from statistics import mean
import click

logger = get_external("npi.mnist")

@click.group()
def cli(): pass

@cli.command()
@click.option("-l", "--log", type=str,
    help="JSONL file to append results to.")
@click.option("--train", type=int, default=100,
    help="Size of the training set.")
@click.option("--test", type=int, default=100,
    help="Size of the test set.")
@click.option("-v", "--verbose", is_flag=True,
    help="Enable verbose output with Sherlog's logging interface.")
@click.option("-t", "--tool", default="torch",
    type=click.Choice(["torch", "sherlog"], case_sensitive=False),
    help="Tool to be evaluated.")
@click.option("-e", "--epochs", type=int, default=1)
@click.option("-l", "--learning-rate", type=float, default=0.01)
def evaluate(log, train, test, verbose, tool, epochs, learning_rate):
    """Evaluate a tool on the NPI MNIST benchmark."""

    if verbose: enable("explanation", "npi.mnist")

    # select the correct interface
    logger.info(f"Loading the {tool} model.")
    model = {
        "torch" : TorchModel,
        "sherlog" : SherlogModel
    }[tool]()

    timer = Timer()

    # train
    logger.info(f"Training with {train} samples...")
    with timer:
        model.fit(sample(train, "train"), epochs=epochs, learning_rate=learning_rate)
    training_time = timer.elapsed
    logger.info(f"Training completed in {training_time} seconds.")

    # collate the results
    result = {
        "seed" : seed(),
        "tool" : tool,
        "train" : train,
        "test" : test,
        "epochs" : epochs,
        "learning_rate" : learning_rate,
        "training_time" : training_time,
    }

    # if a log file is given, write it out
    if log is not None:
        with open(log, 'a') as f:
            f.write(dumps(result))

    console.print(result)

if __name__ == "__main__":
    cli()