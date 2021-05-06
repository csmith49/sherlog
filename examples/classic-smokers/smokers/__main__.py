from json import dumps
from statistics import mean
import click

from sherlog.logs import enable, get_external
from sherlog import console
from sherlog.tooling.instrumentation import seed, Timer
from . import ProblogModel, SherlogModel, sample

logger = get_external("smokers")

@click.group()
def cli():
    pass

@cli.command()
@click.option("-l", "--log", type=str,
    help="JSONL file to append results to.")
@click.option("-s", "--size", type=int,
    help="Size of social graphs; random if not provided.")
@click.option("--train", type=int, default=100,
    help="Size of the training set.")
@click.option("--test", type=int, default=100,
    help="Size of the test set.")
@click.option("-v", "--verbose", is_flag=True,
    help="Enable verbose output with Sherlog's logging interface.")
@click.option("-e", "--epochs", type=int, default=1)
@click.option("-r", "--learning-rate", type=float, default=0.01)
@click.option("-c", "--convergence", is_flag=True)
def sherlog(log, size, train, test, verbose, epochs, learning_rate, convergence):
    """Evaluate Sherlog on the classic Smokers benchmark."""
    if verbose:
        enable("smokers")

    model = SherlogModel()
    timer = Timer()

    logger.info(f"Starting training with {train} samples...")
    with timer:
        if convergence:
            x, y = list(sample(train, size=size)), list(sample(test, size=size))
            training_lls = model.fit(x, y, epochs=epochs, learning_rate=learning_rate)
        else:
            x = list(sample(train, size=size))
            training_lls = model.fit(x, epochs=epochs, learning_rate=learning_rate)
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
    logger.info(f"Evaluation completed: accuracy of {accuracy} with avg. time of {avg_class_time} seconds.")

    result = {
        "seed" : seed(),
        "train" : train,
        "test" : test,
        "tool" : "sherlog",
        "graph_size" : size,
        "epochs" : epochs,
        "learning_rate" : learning_rate,
        "training_time" : training_time,
        "average_log_likelihood" : avg_ll,
        "average_log_likelihood_time" : avg_ll_time,
        "accuracy" : accuracy,
        "average_classification_time" : avg_class_time,
        "training_log_likelihood" : training_lls
    }

    if log is not None:
        with open(log, 'a') as f:
            f.write(dumps(result))

    console.print(result)

@cli.command()
@click.option("-l", "--log", type=str,
    help="JSONL file to append results to.")
@click.option("-s", "--size", type=int,
    help="Size of social graphs; random if not provided.")
@click.option("--train", type=int, default=100,
    help="Size of the training set.")
@click.option("--test", type=int, default=100,
    help="Size of the test set.")
@click.option("-v", "--verbose", is_flag=True,
    help="Enable verbose output with Sherlog's logging interface.")
def problog(log, size, train, test, verbose):
    """Evaluate Problog on the classic Smokers benchmark."""
    if verbose:
        enable("smokers")

    model = ProblogModel()
    timer = Timer()

    logger.info(f"Starting training with {train} samples...")
    with timer:
        training_ll, em_steps = model.fit(list(sample(train, size=size)))
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
    logger.info(f"Evaluation completed: accuracy of {accuracy} with avg. time of {avg_class_time} seconds.")

    result = {
        "seed" : seed(),
        "train" : train,
        "test" : test,
        "tool" : "problog",
        "graph_size" : size,
        "training_time" : training_time,
        "average_log_likelihood" : avg_ll,
        "average_log_likelihood_time" : avg_ll_time,
        "accuracy" : accuracy,
        "average_classification_time" : avg_class_time,
        "training_log_likelihood" : training_ll,
        "em_steps" : em_steps
    }

    if log is not None:
        with open(log, 'a') as f:
            f.write(dumps(result))

    console.print(result)

if __name__ == "__main__":
    cli()