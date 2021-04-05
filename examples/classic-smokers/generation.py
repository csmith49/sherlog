import networkx
import click
from random import random

def flip(weight):
    """Returns true with probability `weight`, and false with probability `1 - weight`.
    
    Parameters
    ----------
    weight : float

    Returns
    -------
    bool
    """
    return random() <= weight

def observe(graph, **kwargs):
    """Observes some concrete instantiation of a social graph.

    Parameters
    ----------
    graph : networkx.Graph

    **kwargs : Dict[str, float]

    Returns
    -------
    Dict[str, Dict[str, Set[int]]]
    """
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


def task(size=10, stress=0.2, influence=0.3, spontaneous=0.1, comorbid=0.3, observed=1.0, evidence=1):
    """Build a social graph with concrete observations.

    Parameters
    ----------
    size : int (default=10)

    stress : float (default=0.2)

    influence : float (default=0.3)

    spontaneous : float (default=0.1)

    comorbid : float (default=0.3)

    observed : float (default=1.0)

    evidence : int (default=1)

    Returns
    -------
    Dict[str, Any]
    """
    
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

def to_sherlog(task):
    """Convert a social graph with concrete observations to a Sherlog problem.

    Parameters
    ----------
    task : Dict[str, Any]

    Returns
    -------
    str
    """
    with open("./programs/smokers.sl", "r") as f:
        source = f.read()
    
    graph_atoms, evidence = [], []

    # convert nodes
    for node in task["graph"]["nodes"]:
        graph_atoms.append(f"person(p_{node}).")
    
    # convert edges
    for (s, d) in task["graph"]["edges"]:
        graph_atoms.append(f"friend(p_{s}, p_{d}).")

    graph = '\n'.join(graph_atoms)

    # convert observations to evidence
    for observation in task["observations"]:
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

def to_problog(task):
    """Convert a social graph with concrete observations to a ProbLog problem.

    Parameters
    ----------
    task : Dict[str, Any]

    Returns
    -------
    (str, str)
    """
    with open("./programs/smokers.pl", "r") as f:
        source = f.read()

    graph_atoms, evidence = [], []

    # convert nodes
    for node in task["graph"]["nodes"]:
        graph_atoms.append(f"person(p_{node}).")
    
    # convert edges
    for (s, d) in task["graph"]["edges"]:
        graph_atoms.append(f"friend(p_{s}, p_{d}).")

    graph = '\n'.join(graph_atoms)

    # convert observations to evidence 
    for observation in task["observations"]:
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
@click.option("-s", "--size", default=10, show_default=True,
    help="Size of generated social graph.")
@click.option("-e", "--evidence", default=1, show_default=True,
    help="Number of i.i.d. observations to generate.")
@click.option("--observed", default=1.0, show_default=True,
    help="Ratio of observed extensional atoms.")
@click.option("--stress", default=0.2, show_default=True,
    help="Parameter: probability a person is stressed.")
@click.option("--influence", default=0.3, show_default=True,
    help="Parameter: probability of influence amongst friends.")
@click.option("--spontaneous", default=0.1, show_default=True,
    help="Parameter: probability of spontaneously developing asthma.")
@click.option("--comorbid", default=0.3, show_default=True,
    help="Parameter: probability of smoking-induced asthma.")
@click.option("-l", "--language", default="sherlog", show_default=True,
    type=click.Choice(["sherlog", "problog"], case_sensitive=False),
    help="The language for which the problem is compiled.")
@click.option("-o","--output", type=click.Path(), help="Output file location.")
def generate(size, evidence, observed, stress, influence, spontaneous, comorbid, language, output):
    # build the problem
    p = task(size=size,
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