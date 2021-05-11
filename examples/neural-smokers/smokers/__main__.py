from . import sample, SherlogModel
from sherlog.logs import enable, get_external
from sherlog import console
from sherlog.tooling.instrumentation import seed, Timer
from sherlog.interface import initialize
from json import dumps
import torch
from statistics import mean
import click

logger = get_external("neural-smokers")

@click.group()
@click.option("-p", "--port", type=int)
def cli(port):
    if port:
        initialize(port=port)
    else:
        initialize()

@cli.command()
@click.option("-l", "--log", type=str, help="JSONL file to append results to.")
@click.option("-s", "--size", type=int, help="Size of graph generated (random if not provided).")
@click.option("-v", "--verbose", is_flag=True, help="Enable vebose output using Sherlog's logging interface.")
@click.option("--train", type=int, default=100)
@click.option("--test", type=int, default=100)
@click.option("-e", "--epochs", type=int, default=1)
@click.option("-r", "--learning-rate", type=float, default=0.01)
def evaluate(log, size, verbose, train, test, epochs, learning_rate):
    """Evaluate Sherlog on the Neural Smokers benchmark."""
    if verbose:
        enable("neural-smokers")

    model = SherlogModel()
    timer = Timer()

    logger.info(f"Starting training with {train} samples...")
    with timer:
        x, y = list(sample(train, size=size)), list(sample(test, size=size))
        training_lls = model.fit(x, y, epochs=epochs, learning_rate=learning_rate)
    training_time = timer.elapsed
    logger.info(f"Training completed in {training_time} seconds.")

    logger.info(f"Evaluating average LL with {test} samples...")
    results = []
    for example in sample(test, size=size):
        with timer:
            ll = model.log_likelihood(example)
        results.append( (ll, timer.elapsed) )
    lls, times = zip(*results)
    avg_ll, avg_ll_time = mean(lls), mean(times)
    logger.info(f"Evaluation completed in {avg_ll_time * test} seconds. Resulting avg. LL: {avg_ll}.")

    logger.info(f"Testing predictive performance with {test} samples...")
    results = []
    for example in sample(test, size=size):
        with timer:
            confidence, ground_truth = model.classification_task(example)
            score = 1.0 if confidence == ground_truth else 0.0
        results.append( (score, timer.elapsed) )
    scores, times = zip(*results)
    accuracy, avg_class_time = mean(scores), mean(times)
    logger.info(f"Evaluation complete: accuracy of {accuracy} with average classification time of {avg_class_time} seconds.")

    result = {
        "seed" : seed(),
        "train" : train,
        "test" : test,
        "graph_size" : size,
        "training_time" : training_time,
        "training_log_likelihood" : training_lls,
        "average_log_likelihood" : avg_ll,
        "average_log_likelihood_time" : avg_ll_time,
        "accuracy" : accuracy,
        "average_classification_time" : avg_class_time
    }

    if log is not None:
        with open(log, 'a') as f:
            f.write(dumps(result))
    
    console.print(result)

if __name__ == "__main__":
    cli()