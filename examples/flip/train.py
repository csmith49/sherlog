import sherlog
from torch.optim import SGD
from alive_progress import alive_bar
from time import sleep

# delay to let the server spin up
print("Letting the SherLog daemon spin up...")
sleep(1)

# load the problem
print("Loading the problem file...")
problem = sherlog.load_problem_file("../examples/flip.sl")

# build the optimizer
optimizer = SGD(problem.parameters(), lr=0.05)

# get the data and iterate
print("Generating the data...")
dataset = list(problem.instances()) * 1000

print("Optimizing parameters...")
with alive_bar(3000) as bar:
    for instance in dataset:
        optimizer.zero_grad()
        loss = instance.loss(num_samples=5)
        loss.backward()
        optimizer.step()
        problem.clamp_parameters()
        bar()

# print the final parameters
print("Learned parameters:")
for name, param in problem._parameters.items():
    print(f"\t{name} -- {param.value:f}")

likelihood = problem.log_likelihood(num_samples=1000)
print(f"Final log-likelihood: {likelihood:f}")