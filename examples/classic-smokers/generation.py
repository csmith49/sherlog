import networkx
import click
from random import random

def flip(weight):
    """Returns true with probability `weight`, and false with probability `1 - weight`."""
    return random() <= weight

def observe(graph, **kwargs):
    # intensional labels
    stress, influence, spontaneous, comorbid = set(), set(), set(), set()

    # build intensional values
    for node in graph:
        if flip(kwargs["stress"]): stress.add(node)
        if flip(kwargs["spontaneous"]): spontaneous.add(node)
        if flip(kwargs["comorbid"]): comorbid.add(node)
    
    for edge in graph.edges():
        if flip(kwargs["influence"]): influence.add(edge)

    # extensional labels
    smokes, asthma = set(), set()

    # build extensional values

    # stress -> smoke
    smokes.update(stress)

    # smoke, influence -> smoke
    # build the influence graph
    influence_graph = networkx.DiGraph()
    influence_graph.add_nodes_from(graph.nodes())
    influence_graph.add_edges_from(influence)

    # and compute transitive closure of influence over smokers
    for node in graph:
        if node in smokes:
            reachable = networkx.algorithms.traversal.dfs_preorder_nodes(influence_graph, node)
            smokes.update(reachable)

    # spontaneous -> asthma
    asthma.update(spontaneous)

    # smoke, comorbid -> asthma
    asthma.update(
        smokes.intersection(comorbid)
    )

    # build observation
    observation = {
        "smokes" : {
            "true" : [],
            "false" : []
        },
        "asthma" : {
            "true" : [],
            "false" : []
        }
    }

    for node in graph:
        # flip for smoking report
        if flip(kwargs["observed"]):
            if node in smokes:
                observation["smokes"]["true"].append(node)
            else:
                observation["smokes"]["false"].append(node)
        
        # flip for asthma report
        if flip(kwargs["observed"]):
            if node in asthma:
                observation["asthma"]["true"].append(node)
            else:
                observation["asthma"]["false"].append(node)
    
    return observation


def problem(size, stress=0.2, influence=0.3, spontaneous=0.1, comorbid=0.3, observed=1.0, evidence=1):
    graph = networkx.generators.directed.scale_free_graph(size)

    result = {
        "graph" : {
            "nodes" : list(graph),
            "edges" : list(graph.edges())
        },
        "observations" : []
    }

    for _ in range(evidence):
        observation = observe(graph,
            stress=stress,
            influence=influence,
            spontaneous=spontaneous,
            comorbid=comorbid,
            observed=observed
        )
        result["observations"].append(observation)

    return result

def to_sherlog(problem):
    with open("./programs/smokers.sl", "r") as f:
        source = f.read()
    
    graph_atoms, evidence = [], []

    # convert nodes
    for node in problem["graph"]["nodes"]:
        graph_atoms.append(f"person(p_{node}).")
    
    # convert edges
    for (s, d) in problem["graph"]["edges"]:
        graph_atoms.append(f"friend(p_{s}, p_{d}).")

    graph = '\n'.join(graph_atoms)

    # convert observations to evidence
    for observation in problem["observations"]:
        atoms = []
        
        for node in observation["smokes"]["true"]:
            atoms.append(f"smokes(p_{node})")
        for node in observation["smokes"]["false"]:
            atoms.append(f"not_smokes(p_{node})")
        for node in observation["asthma"]["true"]:
            atoms.append(f"asthma(p_{node})")
        for node in observation["asthma"]["false"]:
            atoms.append(f"not_asthma(p_{node})")
        
        evidence.append(f"!evidence {', '.join(atoms)}.")
    
    return source + "\n\n" + graph + "\n\n" + '\n'.join(evidence)

def to_problog(problem):
    with open("./programs/smokers.pl", "r") as f:
        source = f.read()

    graph_atoms, evidence = [], []

    # convert nodes
    for node in problem["graph"]["nodes"]:
        graph_atoms.append(f"person(p_{node}).")
    
    # convert edges
    for (s, d) in problem["graph"]["edges"]:
        graph_atoms.append(f"friend(p_{s}, p_{d}).")

    graph = '\n'.join(graph_atoms)

    # convert observations to evidence 
    for observation in problem["observations"]:
        lines = []

        for node in observation["smokes"]["true"]:
            lines.append(f"evidence(smokes(p_{node}), true).")
        for node in observation["smokes"]["false"]:
            lines.append(f"evidence(smokes(p_{node}), false).")
        for node in observation["asthma"]["true"]:
            lines.append(f"evidence(asthma(p_{node}), true).")
        for node in observation["asthma"]["false"]:
            lines.append(f"evidence(asthma(p_{node}), false).")

        evidence.append('\n'.join(lines))

    return source + "\n\n" + graph, "\n---\n".join(evidence)

@click.command()
@click.option("--size", default=10, help="Size of social graph")
@click.option("--evidence", default=1, help="Number of i.i.d. observations to generate")
@click.option("--observed", default=1.0, help="Ratio of observed extensional atoms")
@click.option("--stress", default=0.2, help="Parameter: probability a person is stressed")
@click.option("--influence", default=0.3, help="Parameter: probability of influence amongst friends")
@click.option("--spontaneous", default=0.1, help="Parameter: probability of spontaneously developing asthma")
@click.option("--comorbid", default=0.3, help="Parameter: probability of smoking-induced asthma")
@click.option("--language", default="sherlog",
    type=click.Choice(["sherlog", "problog"], case_sensitive=False),
    help="The language to build the problem for")
@click.option("--output", default="", help="Output file location")
def generate(size, evidence, observed, stress, influence, spontaneous, comorbid, language, output):
    # build the problem
    p = problem(size,
        stress=stress,
        influence=influence,
        spontaneous=spontaneous,
        comorbid=comorbid,
        evidence=evidence
    )

    # split on language
    if language == "sherlog":
        source = to_sherlog(p)
    elif language == "problog":
        source, evidence = to_problog(p)
        source = source + "\n\n" + evidence
    
    # write to output
    if output:
        with open(output, "w") as f:
            f.write(source)
    else:
        print(source)

if __name__ == "__main__":
    generate()