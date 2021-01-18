import networkx
import click
from random import random

def flip(weight):
    return random() <= weight

learnable = [
    # stress
    "!parameter stress : unit",
    "stress_int(P ; bernoulli[stress]) <- person(P)",
    "stress(P) <- stress_int(P, 1.0)",
    
    # influence
    "!parameter influence : unit",
    "influences_int(X, Y ; bernoulli[influence]) <- friend(X, Y)",
    "influences(X, Y) <- influences_int(X, Y, 1.0)",

    # spontaneous cancer
    "!parameter spontaneous : unit",
    "cancer_spontaneous_int(P ; bernoulli[spontaneous]) <- person(P)",
    "cancer_spontaneous(P) <- cancer_spontaneous_int(P, 1.0)",

    # comorbid cancer
    "!parameter comorbid : unit",
    "cancer_smoke_int(P ; bernoulli[comorbid]) <- person(P)",
    "cancer_smoke(P) <- cancer_smoke_int(P, 1.0)"
]

core = [
    "smokes(X) <- stress(X)",
    "smokes(X) <- influences(X, Y), smokes(Y)",
    "cancer(P) <- cancer_spontaneous(P)",
    "cancer(P) <- smokes(P), cancer_smoke(P)",
    "null()"
]

def generate_network(size):
    return networkx.generators.directed.scale_free_graph(size)

def convert_network_to_facts(graph):
    lines = []
    
    # person
    for node in graph:
        line = f"person(p_{node})"
        lines.append(line)

    # friend
    for s, d in graph.edges():
        line = f"friend(p_{s}, p_{d})"
        lines.append(line)

    return lines

def generate_ground_truth(graph, stress=0.2, influence=0.3, spontaneous=0.1, comorbid=0.3):
    # sets for holding results
    results = {
        "stress" : set(),
        "influence" : set(),
        "spontaneous" : set(),
        "comorbid" : set()
    }

    # flip over nodes
    for node in graph:
        if flip(stress): results["stress"].add(node)
        if flip(spontaneous): results["spontaneous"].add(node)
        if flip(comorbid): results["comorbid"].add(node)
    
    # and edges
    for edge in graph.edges():
        if flip(influence): results["influence"].add(edge)

    return results

def generate_observations(graph, ground_truth):
    # sets for holding results
    # some results copied directly from ground truth
    observations = {
        "smokes" : ground_truth["stress"].copy(),
        "cancer" : ground_truth["spontaneous"].copy()
    }

    # check how much influence causes smoking
    influence = set()
    for node in observations["smokes"]:
        influence.update(networkx.algorithms.dag.descendants(graph, node))
    observations["smokes"].update(influence)

    # and check how much smoking causes cancer
    cancer = observations["smokes"].intersection(ground_truth["comorbid"])
    observations["cancer"].update(cancer)

    return observations

def convert_observations_to_evidence(observations):
    atoms = ["null()"]
    for node in observations["smokes"]:
        atom = f"smokes(p_{node})"
        atoms.append(atom)
    for node in observations["cancer"]:
        atom = f"cancer(p_{node})"
        atoms.append(atom)
    return f"!evidence {', '.join(atoms)}"

def generate_problem(size, stress=0.2, influence=0.3, spontaneous=0.1, comorbid=0.3):
    # step 1: make the graph
    g = generate_network(size)
    # step 2: construct the ground truth
    ground_truth = generate_ground_truth(g, stress=0.2, influence=0.3, spontaneous=0.1, comorbid=0.3)
    # step 3: infer observations
    observations = generate_observations(g, ground_truth)
    # step 4: build list of lines
    lines = learnable + core + convert_network_to_facts(g) + [convert_observations_to_evidence(observations)]
    # step 5: build string
    return '\n'.join([f"{line}." for line in lines])

@click.command()
@click.option("--size", default=10, help="Size of social graph")
@click.option("--stress", default=0.2, help="Parameter: smoke-causing stress (unit)")
@click.option("--influence", default=0.3, help="Parameter: smoke-causing peer pressure (unit)")
@click.option("--spontaneous", default=0.1, help="Parameter: rate of spontaneous cancer (unit)")
@click.option("--comorbid", default=0.3, help="Parameter: rate of smoking-induced cancer (unit)")
@click.option("--output", default="", help="Filepath to save output to")
def main(size, stress, influence, spontaneous, comorbid, output):
    problem = generate_problem(size, stress, influence, spontaneous, comorbid)
    print(problem)
    if output != "":
        with open(output, "w") as f:
            f.write(problem)

if __name__ == "__main__":
    main()