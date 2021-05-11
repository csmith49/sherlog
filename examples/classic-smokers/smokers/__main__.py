from json import dumps
from statistics import mean
import torch
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
@click.option("-v", "--verbose", is_flag=True,
    help="Enable verbose output with Sherlog's logging interface.")
@click.option("-e", "--epochs", type=int, default=1)
@click.option("-r", "--learning-rate", type=float, default=0.1)
def overfit(log, size, verbose, epochs, learning_rate):
    """Evaluate overly precise Sherlog on a single example."""
    if verbose:
        enable("smokers")
    
    model = SherlogModel()
    timer = Timer()

    logger.info("Starting graph generation...")
    data = list(sample(1, size))[0]
    logger.info(f"Data: {data}")

    with timer:
        # step 0: concretize the program and training params
        program, evidence = model.program(data)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        training_lls = {}
        # start training
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch}...")
            optimizer.zero_grad()
            objective = torch.tensor(0.0)
            # step 1: sample an explanation propto P(Z)
            logger.info("Generating explanation using MCMC...")
            explanation = program.sample_explanation(
                evidence[0],
                burn_in=20,
                samples=50,
                width=5,
                depth=100,
                attempts=100,
                seeds=5
            )
            # step 2: estimate the ll
            logger.info("Approximating the log-likelihood...")
            log_likelihood = explanation.miser(samples=100).log().mean()
            # step 3: make sure we actually have a gradient
            is_nan = torch.isnan(log_likelihood).any()
            is_inf = torch.isinf(log_likelihood).any()
            if not is_nan and not is_inf:
                objective -= log_likelihood
            
            if objective != 0.0:
                objective.backward()
                optimizer.step()
                model.clamp()

            training_lls[epoch] = log_likelihood.item()
    training_time = timer.elapsed

    logger.info("Testing predictive performance...")
    with timer:
        confidence, ground_truth = model.classification_task(data)
        score = 1.0 if confidence == ground_truth else 0.0
    accuracy, avg_class_time = score, timer.elapsed
    logger.info(f"Evaluation completed: accuracy of {accuracy} with avg. time of {avg_class_time} seconds.")

    result = {
        "seed" : seed(),
        "graph_size" : size,
        "epochs" : epochs,
        "learning_rate" : learning_rate,
        "training_time" : training_time,
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
@click.option("-e", "--epochs", type=int, default=1)
@click.option("-r", "--learning-rate", type=float, default=0.05)
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
            ll = model.log_likelihood(example, width=50, depth=100, seeds=5)
        results.append( (ll, timer.elapsed) )
    lls, times = zip(*results)
    avg_ll, avg_ll_time = mean(lls), mean(times)
    logger.info(f"Evaluation completed in {avg_ll_time * test} seconds. Resulting avg. LL: {avg_ll}.")

    logger.info(f"Testing predictive performance with {test} samples...")
    results = []
    for example in sample(test, size=size):
        with timer:
            confidence, ground_truth = model.classification_task(example, width=5, depth=100, seeds=5)
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