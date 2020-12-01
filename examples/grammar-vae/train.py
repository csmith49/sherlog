import sherlog
from torch.optim import SGD
import time

print("Wait for server to spin up...")
time.sleep(1)

# load the problem
print("Loading the problem file...")
problem = sherlog.load_problem_file("./lists.sl")

# build the optimizer
optimizer = SGD(problem.parameters(), lr=0.01)

# we'll repeat the training for a few epochs
for instance in problem.instances():
    print("Taking a step...")
    optimizer.zero_grad()
    loss = instance.loss()
    print("Loss: ", loss)
    loss.backward()
    optimizer.step()
    problem.clamp_parameters()

print("Saving parameters...")

# save them to disc
problem.save_parameters("/tmp/params.slp")

for name, param in problem._parameters.items():
    print (name, param.value)