import sherlog
from torch.optim import SGD
from itertools import repeat
import time

# delay to let the server spin up
print("Letting the SherLog daemon spin up...")
time.sleep(1)

# load the problem
print("Loading the problem file...")
problem = sherlog.load_problem_file("./examples/flip.sl")

# check the params and evidence we found
print(f"Found: {len(problem.parameters.items())} parameters and {len(problem.evidence)} pieces of data.")

# build the optimizer
optimizer = SGD(problem.trainable_parameters(), lr=0.01)

# we'll repeat the training for a few epochs
for evidence in problem.evidence * 100:
    optimizer.zero_grad()
    story = problem.generative_story(evidence)
    loss = story.loss()
    loss.backward()
    optimizer.step()

# print the final parameters
for name, param in problem.parameters.items():
    print (name, param.value)

print("Saving and loading parameters to test...")

# save them to disc
problem.save_parameters("/tmp/params.slp")

# and, for a sanity check, reset problem and load them back
problem = sherlog.load_problem_file("./examples/flip.sl")
problem.load_parameters("/tmp/params.slp")

for name, param in problem.parameters.items():
    print (name, param.value)