from . import graph, problog
from . import default_parameterization
from sherlog.logs import enable
import click

@click.command()
@click.option("-s", "--size", default=5)
@click.option("--train", default=10)
@click.option("--test", default=10)
@click.option("-v", "--verbose", is_flag=True)
def main(size, train, test, verbose):
    """
    Parameters
    ----------
    size : int
    train : int
    test : int
    verbose : bool
    """
    if verbose: enable("smokers.problog")

    print("Making graphs...")
    train = [graph.Graph(size, default_parameterization) for _ in range(train)]
    test = [graph.Graph(size, default_parameterization) for _ in range(test)]

    print("Training...")
    p = problog.fit(*train)

    print("Evaluating...")
    lls = [problog.log_likelihood(p, g) for g in test]
    avg_ll = sum(lls) / len(lls)

    print(avg_ll)

    print("Classifying...")
    predictions = [problog.classify_asthma(p,g) for g in test]

    print(predictions)

if __name__ == "__main__":
    main()