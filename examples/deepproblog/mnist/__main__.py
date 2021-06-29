from . import mnist, samples, DPLModel, sherlog_model

from sherlog.logs import enable, get_external
from sherlog.tooling.evaluation import evaluate
from sherlog import console
from json import dumps
import click

logger = get_external("examples.mnist")

@click.command()
@click.option("-l", "--log", type=str, help="JSONL file for appending results.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.option("-t", "--tool", default="dpl", type=click.Choice(["dpl", "sherlog"], case_sensitive=False), help="Tool for evaluating.")
@click.option("--train", default=1000, type=int, help="Number of training samples.")
@click.option("--test", default=100, type=int, help="Number of testing samples.")
@click.option("-e", "--epochs", default=1, type=int, help="Number of training epochs.")
def cli(log, verbose, tool, train, test, epochs):
    """CLI for the MNIST benchmark."""
    if verbose:
        enable("examples.mnist")

    # initialize the necessary model
    logger.info(f"Loading the {tool} model...")
    model = {
        "dpl" : DPLModel(mnist),
        "sherlog" : sherlog_model
    }[tool]

    # evaluate performance
    logger.info(f"Beginning evaluation...")
    results = evaluate(
        model,
        samples,
        train_size=train,
        test_size=test,
        fit_kwargs={
            "epochs" : epochs,
            "batch_size" : 1,
            "explanations" : 1,
            "samples" : 500
        },
        log_prob_kwargs={}
    )

    # append some extra info to the results
    results["tool"] = tool

    # process the final results
    if log is not None:
        with open(log, 'a') as f:
            f.write(dumps(results))
    
    # and display, for good measure
    console.print(results)

if __name__ == "__main__":
    cli()