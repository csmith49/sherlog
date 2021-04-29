from . import sample, SherlogModel
from sherlog.logs import enable, get_external
from sherlog import console
from sherlog.tooling.instrumentation import seed, Timer
from json import dumps
from statistics import mean
import click

logger = get_external("neural-smokers")

@click.group()
def cli(): pass

@cli.command()
@click.option("-l", "--log", type=str, help="JSONL file to append results to.")
@click.option("-s", "--size", type=int, help="Size of graph generated (random if not provided).")
@click.option("-v", "--verbose", is_flag=True, help="Enable vebose output using Sherlog's logging interface.")
@click.option("--train", type=int, default=100)
@click.option("--test", type=int, default=100)
@click.option("-e", "--epochs", type=int, default=1)
@click.option("-r", "--learning-rate", type=float, default=0.1)
def evaluate(log, size, verbose, train, test, epochs, learning_rate):
    """Evaluate Sherlog on the Neural Smokers benchmark."""
    if verbose: enable("neural-smokers")

    model = SherlogModel()
    timer = Timer()

    logger.info(f"Starting training with {train} samples...")
    with timer:
        model.fit(sample(train, size=size), epochs=epochs)
    training_time = timer.elapsed

    console.print(training_time)

if __name__ == "__main__":
    cli()