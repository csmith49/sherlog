from . import graph, problog
from . import default_parameterization
import click

@click.command()
@click.option("-s", "--size", default=5)
@click.option("--train", default=10)
@click.option("--test", default=10)
def main(size, train, test):

    print("Making graphs...")
    train = [graph.Graph(size, default_parameterization) for _ in range(train)]
    test = [graph.Graph(size, default_parameterization) for _ in range(test)]

    print("Training...")
    p = problog.fit(*train)

    print("Evaluating...")
    lls = [problog.log_likelihood(p, g) for g in test]
    avg_ll = sum(lls) / len(lls)

    print(avg_ll)

if __name__ == "__main__":
    main()