import sherlog
from torch.optim import SGD
import time

# delay to let the server spin up
print("Letting the SherLog daemon spin up...")
time.sleep(1)

# load the problem
print("Loading the problem file...")
problem = sherlog.load_problem_file("./examples/flip.sl")

# build the optimizer
optimizer = SGD(problem.parameters(), lr=0.1)

# we'll repeat the training for a few epochs
for story, context in problem.stories():
    optimizer.zero_grad()
    context = story.run(context)
    loss = story.loss(context)
    loss.backward()
    optimizer.step()

    for name, param in problem._parameters.items():
        print (name, param.value, param.value.grad)

    problem.clamp_parameters()

# print the final parameters
for name, param in problem._parameters.items():
    print (name, param.value)

print("Saving and loading parameters to test...")

# save them to disc
problem.save_parameters("/tmp/params.slp")

# and, for a sanity check, reset problem and load them back
problem = sherlog.load_problem_file("./examples/flip.sl")
problem.load_parameters("/tmp/params.slp")

for name, param in problem._parameters.items():
    print (name, param.value)