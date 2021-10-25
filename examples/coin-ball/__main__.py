import click

from sherlog.inference import minibatch
from sherlog.interface import initialize
from sherlog.interface.instrumentation import minotaur

from torch.nn.functional import softmax

from rich import print

from .data import sample, embedder, random_coin, random_color
from .program import Program

@click.command()
@click.option("-t", "--train", default=100, type=int, help="Number of training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Training epochs.")
@click.option("-l", "--learning-rate", default=1e-4, type=float, help="Learning rate.")
@click.option("-i", "--instrumentation", type=str, help="Instrumentation log destination.")
def cli(train, batch_size, epochs, learning_rate, instrumentation):
    # set up instrumentation before initializing sherlog
    if instrumentation: minotaur.add_filepath_handler(instrumentation)

    # initialize the query server
    initialize(port=8007)

    minotaur["train"] = train
    minotaur["batch size"] = batch_size
    minotaur["epochs"] = epochs
    minotaur["learning rate"] = learning_rate

    # set up ground truth
    urn_one_weights = [0.7, 0.3]
    urn_two_weights = [0.4, 0.3, 0.3]

    # initialize the learning task
    data = [sample(
        coin_weights=[0.5, 0.5],
        urn_one_weights=urn_one_weights,
        urn_two_weights=urn_two_weights
    ) for _ in range(train)]

    program = Program()
    optimizer = program.optimizer(learning_rate)

    # iterate over data and optimize
    for batch in minibatch(data, batch_size, epochs=epochs):
        with minotaur("batch"):
            optimizer.maximize(*embedder.embed_all(batch.data))
            batch_loss = optimizer.optimize()

            print(f"Batch {batch.index}:{batch.epoch} loss: {batch_loss.item()}")

            minotaur["batch"] = batch.index
            minotaur["epoch"] = batch.epoch

            with minotaur("evaluate"):
                # show convergence of program parameters
                print(f"\tUrn one convergence: {program.urn_one_convergence(urn_one_weights)}")
                print(f"\tUrn two convergence: {program.urn_two_convergence(urn_two_weights)}")

                # and test accuracy of modules
                coins = [random_coin() for _ in range(1000)]
                print(f"\tDownstream coin_nn performance: {program.coin_nn_performance(coins)}")
                
                colors = [random_color() for _ in range(1000)]
                print(f"\tDownstream color_nn performance: {program.color_nn_performance(colors)}")
                

    for parameter in program.program._parameters:
        name, value = parameter.name, softmax(parameter.value, dim=0).tolist()
        minotaur[name] = value
        print(f"{name} : {value}")

if __name__ == "__main__":
    cli()