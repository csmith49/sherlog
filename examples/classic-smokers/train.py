import sherlog
import click
import generation
from torch.optim import SGD, Adam
from time import sleep
from sherlog.instrumentation import Instrumenter, seed
from sherlog import console
from rich.progress import track

@click.command()
@click.option("--epochs", default=1000, help="Number of training epochs")
@click.option("--optimizer", default="sgd",
    type=click.Choice(["sgd", "adam"], case_sensitive=False),
    help="Torch optimizer used for tuning Sherlog")
@click.option("--learning-rate", default=0.01, help="Learning rate for the Torch optimizer")
@click.option("--log/--no-log", default=False, help="Enables/disables recording of results")
@click.option("--observed", default=1.0, help="Ratio of extensional atoms observed")
@click.option("--evidence", default=1, help="Number of i.i.d. observations to optimize w.r.t")
@click.option("--size", default=5, help="Size of generated social graph")
@click.option("--stress", default=0.2, help="Parameter: probability a person is stressed")
@click.option("--influence", default=0.3, help="Parameter: probability of influence amongst friends")
@click.option("--spontaneous", default=0.1, help="Parameter: probability of spontaneously developing asthma")
@click.option("--comorbid", default=0.3, help="Parameter: probability of smoking-induced asthma")
def main(**kwargs):

    # build abstract problem to convert to sherlog / problog programs
    with console.status("Generating random social graph..."):
        abstract_problem = generation.problem(
            kwargs["size"],
            stress=kwargs["stress"],
            influence=kwargs["influence"],
            spontaneous=kwargs["spontaneous"],
            comorbid=kwargs["comorbid"],
            observed=kwargs["observed"],
            evidence=kwargs["evidence"]
        )
    
    # instrumentation
    instrumenter = Instrumenter("classic-smokers-results.jsonl", context={
        "optimizer" : kwargs["optimizer"],
        "learning-rate" : kwargs["learning_rate"],
        "observed" : kwargs["observed"],
        "evidence" : kwargs["evidence"],
        "size" : kwargs["size"],
        "gt:stress" : kwargs["stress"],
        "gt:influence" : kwargs["influence"],
        "gt:spontaneous" : kwargs["spontaneous"],
        "gt:comorbid" : kwargs["comorbid"],
        "seed" : seed()
    })

    # SHERLOG TRAINING
    problem = sherlog.problem.loads(generation.to_sherlog(abstract_problem))

    optim = {
        "sgd" : SGD,
        "adam" : Adam
    }[kwargs["optimizer"]](problem.parameters(), lr=kwargs["learning_rate"])

    optimizer = sherlog.inference.Optimizer(problem, optim)

    for epoch in track(range(kwargs["epochs"]), description="Training Sherlog"):
        stories = list(problem.stories(depth=100, width=10))
        with optimizer as o:
            for i, story in enumerate(stories):
                o.maximize(story.objective(index=i))
                output = {
                    "model" : [str(ass) for ass in story.model.assignments],
                    "meet" : story.meet.mapping,
                    "avoid" : story.avoid.mapping
                }
                console.print(output)

        if epoch % 100 == 0:
            console.rule(f"Epoch {epoch}")
            console.print(len(stories))
            console.print(problem._parameters.items())
            likelihood = problem.log_likelihood(num_samples=100)
            instrumenter.emit(
                tool="sherlog",
                likelihood=likelihood.item(),
                step=epoch,
                stress=problem._parameters["stress"].value.item(),
                influence=problem._parameters["influence"].value.item(),
                spontaneous=problem._parameters["spontaneous"].value.item(),
                comorbid=problem._parameters["comorbid"].value.item()
            )
    
    print("Learned parameters:")
    for name, param in problem._parameters.items():
        print(f"\t{name} -- {param.value:f}")
    
    likelihood = problem.log_likelihood(num_samples=100)
    print(f"Final log-likelihood: {likelihood:f}")

    if kwargs["log"]: instrumenter.flush()

if __name__ == "__main__":
    main()