import generation
import sherlog
from argparse import ArgumentParser
from hashids import Hashids
from random import randint
from torch.optim import SGD, Adam
from alive_progress import alive_bar
from time import sleep
from sherlog.instrumentation import Instrumenter
import storch

# command-line arguments
parser = ArgumentParser("Smoker - Sherlog Evaluation Script")
parser.add_argument("--epochs", default=700, help="Number of training epochs")
parser.add_argument("--resolution", default=100, help="Instrumentation resolution")
parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"], help="Optimization strategy")
parser.add_argument("--learning-rate", default=0.01, help="Optimizer learning rate")
parser.add_argument("--log", action="store_true", help="Enables recording of results")
parser.add_argument("--size", default=10, type=int, help="Generated network size")
parser.add_argument("--stress", default=0.2, type=float, help="Parameter: smoke-causing stress (unit)")
parser.add_argument("--influence", default=0.3, type=float, help="Parameter: smoke-causing peer pressure (unit)")
parser.add_argument("--spontaneous", default=0.1, type=float, help="Parameter: rate of spontaneous cancer (unit)")
parser.add_argument("--comorbid", default=0.3, type=float, help="Parameters: rate of smoking-induced cancer (unit)")

args = parser.parse_args()

# delay to let the server spin up
print("Spinning up the server...")
sleep(1)

# initializing a bunch of stuff
hashids = Hashids()
seed = hashids.encode(randint(0, 100000))

print("Building the problem...")
g = generation.generate_problem(
        args.size,
        stress=args.stress,
        influence=args.influence,
        spontaneous=args.spontaneous,
        comorbid=args.comorbid
    )
print(g)
problem = sherlog.problem.loads(g)

print("Initializing the optimizer and instrumentation...")
optim = {
    "sgd" : SGD,
    "adam" : Adam
}[args.optimizer](problem.parameters(), lr=args.learning_rate)

instrumenter = Instrumenter("smoker-results.jsonl", context={
    "optimizer" : args.optimizer,
    "learning-rate" : args.learning_rate,
    "seed" : seed,
    "size" : args.size,
    "stress_gt" : args.stress,
    "influence_gt" : args.influence,
    "spontaneous_gt" : args.spontaneous,
    "comorbid_gt" : args.comorbid
})

print("Starting training...")

# start the training
with alive_bar(args.epochs) as bar:
    for i in range(args.epochs):
        for stories in problem.stories():
            # compute gradients
            optim.zero_grad()
            loss = problem.objective(stories, k=5)
            storch.backward()
            optim.step()
            problem.clamp_parameters()
        
        # update the bar
        bar()

        # generate some statistics
        if i % args.resolution == 0:
            likelihood = problem.log_likelihood(num_samples=1)
            instrumenter.emit(
                likelihood=likelihood.item(),
                step=i,
                stress=problem._parameters["stress"].value.item(),
                influence=problem._parameters["influence"].value.item(),
                spontaneous=problem._parameters["spontaneous"].value.item(),
                comorbid=problem._parameters["comorbid"].value.item()
            )

# print the final parameters
print("Learned parameters:")
for name, param in problem._parameters.items():
    print(f"\t{name} -- {param.value:f}")

likelihood = problem.log_likelihood(num_samples=1)
print(f"Final log-likelihood: {likelihood:f}")

if args.log: instrumenter.flush()