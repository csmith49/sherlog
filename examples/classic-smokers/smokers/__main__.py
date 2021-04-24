from . import problog_interface, sherlog_interface, default_parameterization, dict_product
from sherlog.logs import enable, get_external
from sherlog import console
from sherlog.tooling.instrumentation import seed
from json import dumps
from typing import Optional
import click

logger = get_external("smokers")

@click.group()
def cli(): pass

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
@click.option("-t", "--tool", default="problog",
    type=click.Choice(["problog", "sherlog"], case_sensitive=False),
    help="Tool to be evaluated.")
def evaluate(log, size, train, test, verbose, tool):
    """Evaluate a tool on the classic Smokers benchmark."""
    
    if verbose: enable("smokers")

    # select the correct interface
    logger.info(f"Loading the {tool} interface.")
    interface = {
        "problog" : problog_interface,
        "sherlog" : sherlog_interface
    }[tool]

    # train
    logger.info(f"Training with {train} samples...")
    training_time = interface.fit(train, graph_size=size)
    logger.info(f"Training completed in {training_time} seconds.")

    # evaluate
    logger.info(f"Evaluating the average log-likelihood with {test} samples...")
    avg_ll, avg_ll_time = interface.average_log_likelihood(test, graph_size=size)
    logger.info(f"Evaluation complete: average log-likelihood of {avg_ll} with average marginal time of {avg_ll_time} seconds.")

    # test classification performance
    logger.info(f"Evaluating the asthma prediction accuracy with {test} samples...")
    accuracy, avg_class_time = interface.classification_accuracy(test, graph_size=size)
    logger.info(f"Evaluation complete: accuracy of {accuracy} with average classification time of {avg_class_time} seconds.")

    # collate the results
    result = {
        "seed" : seed(),
        "tool" : tool,
        "train" : train,
        "test" : test,
        "graph_size" : size,
        "training_time" : training_time,
        "average_log_likelihood" : avg_ll,
        "average_log_likelihood_time" : avg_ll_time,
        "gt_stress" : default_parameterization.stress,
        "stress" : interface._parameterization.stress,
        "gt_comorbid" : default_parameterization.comorbid,
        "comorbid" : interface._parameterization.comorbid,
        "gt_spontaneous" : default_parameterization.spontaneous,
        "spontaneous" : interface._parameterization.spontaneous,
        "gt_influence" : default_parameterization.influence,
        "influence" : interface._parameterization.influence,
        "accuracy" : accuracy,
        "average_classification_time" : avg_class_time
    }

    # if a log file is given, append the results
    if log is not None:
        with open(log, 'a') as f:
            f.write(dumps(result))
    
    console.print(result)

@cli.command()
@click.option("-l", "--log", type=str,
    help="JSONL file to append results to.")
@click.option("-v", "--verbose", is_flag=True,
    help="Enables verbose output using Sherlog's logging interface.")
@click.option("--train", type=int, default=10,
    help="Size of the training set.")
@click.option("--test", type=int, default=10,
    help="Size of the test set.")
def tune(log, verbose, train, test):
    """Evaluate Sherlog on a broad range of hyperparameters."""

    if verbose: enable("smokers")

    # load the sherlog interface
    logger.info(f"Loading the interface.")
    interface = sherlog_interface

    # set up the hyperparameters to search through
    kwargs_gen = dict_product(
        epochs=[1, 10, 50, 100],
        learning_rate=[0.5, 0.1, 0.05, 0.01, 0.005, 0.001],
        stories=[1, 5, 10, 50],
        samples=[1, 10, 100, 1000],
        width=[1, 5, 10, 50, 100],
        depth=[10, 50, 100],
        attempts=[1, 50, 100],
        seeds=[1, 5, 10, 50],
        graph_size=[3, 5, 10]
    )

    for kwargs in kwargs_gen:
        logger.info(f"Starting tuning run with parameterization: {kwargs}")
        logger.info(f"Training with {train} samples...")
        training_time = interface.fit(train, graph_size=kwargs["graph_size"], fit_kwargs=kwargs)
        logger.info(f"Training completed in {training_time} seconds.")

        # evaluate
        logger.info(f"Evaluating the average log-likelihood with {test} samples...")
        avg_ll, avg_ll_time = interface.average_log_likelihood(test, graph_size=kwargs["graph_size"], ll_kwargs=kwargs)
        logger.info(f"Evaluation complete: average log-likelihood of {avg_ll} with average marginal time of {avg_ll_time} seconds.")

        # test classification performance
        logger.info(f"Evaluating the asthma prediction accuracy with {test} samples...")
        accuracy, avg_class_time = interface.classification_accuracy(test, graph_size=kwargs["graph_size"], class_kwargs=kwargs)
        logger.info(f"Evaluation complete: accuracy of {accuracy} with average classification time of {avg_class_time} seconds.")

        # collate the results
        result = {
            "seed" : seed(),
            "train" : train,
            "test" : test,
            "training_time" : training_time,
            "average_log_likelihood" : avg_ll,
            "average_log_likelihood_time" : avg_ll_time,
            "gt_stress" : default_parameterization.stress,
            "stress" : interface._parameterization.stress,
            "gt_comorbid" : default_parameterization.comorbid,
            "comorbid" : interface._parameterization.comorbid,
            "gt_spontaneous" : default_parameterization.spontaneous,
            "spontaneous" : interface._parameterization.spontaneous,
            "gt_influence" : default_parameterization.influence,
            "influence" : interface._parameterization.influence,
            "accuracy" : accuracy,
            "average_classification_time" : avg_class_time
        }
        result.update(kwargs)

        console.print(result)

        # if a log file is given, append the results
        if log is not None:
            with open(log, 'a') as f:
                f.write(f"{dumps(result)}\n")

if __name__ == "__main__":
    cli()