import sherlog
import click
import generation

from torch.optim import SGD, Adam
from rich.progress import track

def problem_instance(**kwargs):
    """Build a Sherlog problem instance.

    Parameters
    ----------
    **kwargs : Dict[str, Any]

    Returns
    -------
    sherlog.Problem
    """
    task = generation.task(**kwargs)
    source = generation.to_sherlog(task)
    return sherlog.problem.loads(source)

@click.command()
@click.option("-i", "--instrument", type=click.Path(),
    help="Destination file for instrumentation logs.")
@click.option("-r", "--resolution", default=50,
    help="Instrumentation resolution (in epochs).")
@click.option("-e", "--epochs", default=300, help="Number of training epochs.")
@click.option("-o", "--optimizer", default="sgd", show_default=True,
    type=click.Choice(["sgd", "adam"], case_sensitive=False),
    help="Optimization strategy for training parameters.")
@click.option("-l", "--learning-rate", default=0.01, show_default=True,
    help="Optimizer learning rate.")
@click.option("-s", "--samples", default=100, show_default=True,
    help="Samples-per-explanatioon in gradient estimation.")
@click.option("--size", default=10, show_default=True,
    help="Size of generated social graph.")
@click.option("--evidence", default=1, show_default=True,
    help="Number of i.i.d. observations generated.")
@click.option("--observed", default=1.0, show_default=True,
    help="Ratio of observed extensional atoms.")
@click.option("--stress", default=0.2, show_default=True,
    help="Parameter: probability a person is stressed.")
@click.option("--influence", default=0.3, show_default=True,
    help="Parameter: probability of influence amongst friends.")
@click.option("--spontaneous", default=0.1, show_default=True,
    help="Parameter: probability of spontaneously developing asthma.")
@click.option("--comorbid", default=0.3, show_default=True,
    help="Parameter: probability of smoking-induced asthma.")
def evaluate(instrument, resolution, epochs, optimizer, learning_rate, samples, size, evidence, observed, stress, influence, spontaneous, comorbid):
    # build the problem
    problem = problem_instance(
        size=size,
        observed=observed,
        evidence=evidence,
        stress=stress,
        influence=influence,
        spontaneous=spontaneous,
        comorbid=comorbid
    )

    # build the instrumenter
    instrumenter = sherlog.instrumentation.Instrumenter(instrument, context={
        "seed" : sherlog.instrumentation.seed(),
        "benchmark" : "classic-smokers.sl",
        "epochs" : epochs,
        "optimizer" : optimizer,
        "learning rate" : learning_rate,
        "samples" : samples,
        "size" : size,
        "observed" : observed,
        "evidence" : evidence,
        "gt:stress" : stress,
        "gt:influence" : influence,
        "gt:spontaneous" : spontaneous,
        "gt:comorbid" : comorbid
    })

    # build the optimizer
    opt = {
        "sgd" : SGD,
        "adam" : Adam
    }[optimizer](problem.parameters(), lr=learning_rate)

    optimizer = sherlog.inference.Optimizer(problem, opt)

    # now start training
    for epoch in track(range(epochs), description="Training"):
        with optimizer as o:
            for obj in problem.objectives(epoch=epoch, samples=samples):
                o.maximize(obj)

        if epoch % resolution == 0:
            log = {"epoch" : epoch}
            for k, v in problem.parameter_map.items():
                log[k] = v
                log[f"{k} grad"] = v.grad
            instrumenter.emit(**log)

    if instrument: instrumenter.flush()

    # print the values of the learned parameters
    print("MLE Results")
    for name, parameter in problem.parameter_map.items():
        print(f"{name}: {parameter:f}")

if __name__ == "__main__":
    evaluate()