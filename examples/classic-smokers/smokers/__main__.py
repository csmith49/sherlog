from . import problog_interface, sherlog_interface, default_parameterization
from sherlog.logs import enable, get_external
from sherlog import console
from sherlog.tooling.instrumentation import seed
from json import dumps
import click

logger = get_external("smokers")

@click.command()
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
    logger.info(f"Evaluation complete: average log-likelihood of {avg_ll} computed in {avg_ll_time} seconds.")

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
        "influence" : interface._parameterization.influence
    }

    # if a log file is given, append the results
    if log is not None:
        with open(log, 'a') as f:
            f.write(dumps(result))
    
    console.print(result)

if __name__ == "__main__":
    evaluate()