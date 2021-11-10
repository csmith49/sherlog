"""
"""

import click

from sherlog.program import loads
from sherlog.interface import print, initialize, minotaur
from sherlog.inference import minibatch, Optimizer, FunctionalEmbedding

from torch import tensor
from random import random

SOURCE = \
"""
!parameter earthquake_sensitivity : unit.
!parameter burglary_sensitivity : unit.

unit(H, C) <- house(H, C).
unit(B, C) <- business(B, C).

earthquake(C; bernoulli[0.1]) <- city(C, _).
burglary(X; bernoulli[R]) <- unit(X, C), city(C, R).

alarm(X; {off, on} <~ bernoulli[earthquake_sensitivity]) <- unit(X, C), earthquake(C, 1.0).
alarm(X; {off, on} <~ bernoulli[burglary_sensitivity]) <- unit(X, _), burglary(X, 1.0).
alarm(X, off) <- unit(X, C), burglary(X, 0.0), earthquake(C, 0.0).
"""

# TABLES FOR EXTENSIONAL RELATIONS

CITY = [
    ("houston", 7.44 / 1000),
    ("austin", 4.48 / 1000),
    ("dallas", 3.41 / 1000),
    ("fort_worth", 4.37 / 1000),
    ("san_antonio", 5.29 / 1000),
    ("el_paso", 1.58 / 1000)
]

HOUSE = [
    ("house61", "houston"),
    ("house62", "houston"),
    ("house63", "houston"),
    ("house64", "houston"),
    ("house65", "houston"),
    ("house51", "austin"),
    ("house52", "austin"),
    ("house53", "austin"),
    ("house54", "austin"),
    ("house55", "austin"),
    ("house41", "dallas"),
    ("house42", "dallas"),
    ("house43", "dallas"),
    ("house44", "dallas"),
    ("house44", "dallas"),
    ("house31", "fort_worth"),
    ("house32", "fort_worth"),
    ("house33", "fort_worth"),
    ("house34", "fort_worth"),
    ("house35", "fort_worth"),
    ("house21", "san_antonio"),
    ("house22", "san_antonio"),
    ("house23", "san_antonio"),
    ("house24", "san_antonio"),
    ("house25", "san_antonio"),
    ("house11", "el_paso"),
    ("house12", "el_paso"),
    ("house13", "el_paso"),
    ("house14", "el_paso"),
    ("house15", "el_paso")
]

BUSINESS = [
    ("business61", "houston"),
    ("business62", "houston"),
    ("business63", "houston"),
    ("business51", "austin"),
    ("business52", "austin"),
    ("business53", "austin"),
    ("business41", "dallas"),
    ("business42", "dallas"),
    ("business43", "dallas"),
    ("business31", "fort_worth"),
    ("business32", "fort_worth"),
    ("business33", "fort_worth"),
    ("business21", "san_antonio"),
    ("business22", "san_antonio"),
    ("business23", "san_antonio"),
    ("business11", "el_paso"),
    ("business12", "el_paso"),
    ("business13", "el_paso"),
]

def table(name : str, rows):
    """Convert a table into a string representation amenable to parsing via Sherlog."""
    def line_gen():
        for row in rows:
            yield f"{name}({', '.join(str(item) for item in row)})."
    return "\n" + "\n".join(line_gen())

# GROUND TRUTH SEMANTICS

def flip(p : float): return random() <= p

def sample(city, house, business, earthquake_rate : float = 0.1, earthquake_sensitivity : float = 0.6, burglary_sensitivity : float = 0.9):
    # who has a sensing unit?
    units = house + business
    
    # check which cities have an earthquake
    earthquake = {city : flip(earthquake_rate) for city, _ in city}
    
    # check which places have a burglary
    burglary_rate = {city : rate for city, rate in city}
    burglary = {unit : flip(burglary_rate[city]) for unit, city in units}
    
    # check which units have alarms going off
    earthquake_triggers = {unit : (flip(earthquake_sensitivity) if earthquake[city] else False) for unit, city in units}
    burglary_triggers = {unit : (flip(burglary_sensitivity) if burglary[unit] else False) for unit, _ in units}
    
    # and merge all that info together
    alarms = {unit : (earthquake_triggers[unit] or burglary_triggers[unit]) for unit, _ in units}
    
    # our ground-truth is this *entire* process, so we can get fine-grained info during training
    return {
        "unit" : units,
        "earthquake" : earthquake,
        "burglary" : burglary,
        "earthquake_trigger" : earthquake_triggers,
        "burglary_trigger" : burglary_triggers,
        "alarm" : alarms
    }

def evidence_of_alarm(unit : str, alarm : bool) -> str:
    return f"alarm({unit}, {'on' if alarm else 'off'})"

def embed(alarms) -> str:
    result = ", ".join(evidence_of_alarm(unit, alarm) for unit, alarm in alarms.items())
    return result

@click.command()
@click.option("-t", "--train", default=100, type=int, help="Number of i.i.d. training samples.")
@click.option("-b", "--batch-size", default=10, type=int, help="Training minibatch size.")
@click.option("-e", "--epochs", default=10, type=int, help="Number of training epochs.")
@click.option("-l", "--learning-rate", type=float, default=1e-4, help="Optimization learning rate.")
@click.option("-f", "--forcing", is_flag=True, help="Enable/disable explanation forcing.")
@click.option("-c", "--caching", is_flag=True, help="Enable/disable explanation sampling caching.")
@click.option("-s", "--samples", default=1, type=int, help="Number of per-explanation executions.")
@click.option("-i", "--instrumentation", type=str, help="Instrumentation log destination.")
def cli(**kwargs):
    """Learn the sensitivities of intrusion-detection units."""

    # initialize
    print("Initializing...")
    initialize(port=8007, instrumentation=kwargs["instrumentation"])

    minotaur.enter("burglary")
    minotaur["train"] = kwargs["train"]
    minotaur["batch-size"] = kwargs["batch_size"]
    minotaur["epochs"] = kwargs["epochs"]

    # load the program
    print("Loading the program...")
    program, _ = loads(SOURCE + table("city", CITY) + table("house", HOUSE) + table("business", BUSINESS))

    # load the data
    print(f"Generating {kwargs['train']} training points...")
    data = [sample(CITY, HOUSE, BUSINESS,
        earthquake_rate=0.1,
        earthquake_sensitivity=0.6,
        burglary_sensitivity=0.9
    ) for _ in range(kwargs["train"])]
    embedder = FunctionalEmbedding(evidence=lambda sample: embed(sample["alarm"]))

    # build the optimizer
    print(f"Initializing the optimizer with a learning rate of {kwargs['learning_rate']}...")
    optimizer = Optimizer(
        program,
        learning_rate=kwargs["learning_rate"],
        samples=kwargs["samples"],
        force=kwargs["forcing"],
        cache=kwargs["caching"]
    )

    # and optimize
    old_batch_loss = tensor(0.0)
    for batch in minibatch(data, kwargs["batch_size"], epochs=kwargs["epochs"]):
        with minotaur("batch"):
            # frame
            print(f"\nBatch {batch.index:03d} in Epoch {batch.epoch:03d}")
            minotaur["batch"] = batch.index
            minotaur["epoch"] = batch.epoch

            # get ground-truth stuff

            # okay, let's optimize
            optimizer.maximize(*embedder.embed_all(batch.data))
            batch_loss = optimizer.optimize()
            print(f"Batch loss: {batch_loss:.3f} (Δ={old_batch_loss - batch_loss:.3f})")

            # what are the parameters doing?
            print("Parameter summary:")

            earthquake_sensitivity = program.parameter("earthquake_sensitivity")
            print(f"earthquake_sensitivity={earthquake_sensitivity.item():.3f}, ∇p={earthquake_sensitivity.grad.item():.3f}, error=±{abs(earthquake_sensitivity.item() - 0.0):.3f}")
            minotaur["earthquake_sensitivity"] = earthquake_sensitivity.item()
            minotaur["earthquake_sensitivity-grad"] = earthquake_sensitivity.grad.item()

            burglary_sensitivity = program.parameter("burglary_sensitivity")
            print(f"burglary_sensitivity={burglary_sensitivity.item():.3f}, ∇p={burglary_sensitivity.grad.item():.3f}, error=±{abs(burglary_sensitivity.item() - 0.0):.3f}")
            minotaur["burglary_sensitivity"] = burglary_sensitivity.item()
            minotaur["burglary_sensitivity-grad"] = burglary_sensitivity.grad.item()

            # for getting the batch loss delta next time
            old_batch_loss = batch_loss

    minotaur.exit()

if __name__ == "__main__":
    cli()