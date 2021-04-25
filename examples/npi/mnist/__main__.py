from . import sample, DPLModel, SherlogModel
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
@click.option("-t", "--tool", default="deepproblog",
    type=click.Choice(["deepproblog", "sherlog"], case_sensitive=False),
    help="Tool to be evaluated.")
@click.option("-e", "--epochs", type=int, default=1)
@click.option("-l", "--learning-rate", type=float, default=0.1)
def evaluate(log, train, test, verbose, tool, epochs, learning_rate):
    """Evaluate a tool on the NPI MNIST benchmark."""

    if verbose: enable("npi.mnist")

    # select the correct interface
    logger.info(f"Loading the {tool} model.")
    model = {
        "deepproblog" : DPLModel,
        "sherlog" : SherlogModel
    }[tool]()

    timer = Timer()

    # train
    logger.info(f"Training with {train} samples...")
    with timer:
        model.fit(sample(train, "train"), epochs=epochs, learning_rate=learning_rate)
    training_time = timer.elapsed
    logger.info(f"Training completed in {training_time} seconds.")

    # avg ll
    logger.info(f"Computing average log-likelihood with {test} samples...")
    results = []
    for test_example in sample(test, "test"):
        with timer:
            ll = model.log_likelihood(test_example)
        results.append( (ll, timer.elapsed) )
    lls, times = zip(*results)
    avg_ll, avg_ll_time = mean(lls), mean(times)
    logger.info(f"Evaluation completed in {avg_ll_time * test} seconds. Resulting avg. LL: {avg_ll}.")

    # test
    logger.info(f"Testing predictive performance with {test} samples...")
    results = []
    for test_example in sample(test, "test"):
        with timer:
            acc = model.completion(test_example)
        results.append( (acc, timer.elapsed) )
    accuracies, times = zip(*results)
    accuracy, avg_class_time = mean(accuracies), mean(times)
    logger.info(f"Evaluation completed in {avg_class_time * test} seconds. Resulting accuracy: {accuracy}.")

    # collate the results
    result = {
        "seed" : seed(),
        "tool" : tool,
        "train" : train,
        "test" : test,
        "epochs" : epochs,
        "average_log_likelihood" : avg_ll,
        "average_log_likelihood_time":  avg_ll_time,
        "learning_rate" : learning_rate,
        "training_time" : training_time,
        "accuracy" : accuracy,
        "average_classification_time" : avg_class_time
    }

    # if a log file is given, write it out
    if log is not None:
        with open(log, 'a') as f:
            f.write(dumps(result))

    console.print(result)

if __name__ == "__main__":
    cli()